require("dotenv").config();
const express = require("express");
const cors = require("cors");
const admin = require("firebase-admin");
const { GoogleGenerativeAI } = require("@google/generative-ai");

// Initialize Firebase Admin SDK
// Render can load service account from GOOGLE_APPLICATION_CREDENTIALS or FIREBASE_SERVICE_ACCOUNT env var.
// If FIREBASE_SERVICE_ACCOUNT is provided as a JSON string, parse it.
if (process.env.FIREBASE_SERVICE_ACCOUNT) {
  try {
    const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
    admin.initializeApp({
      credential: admin.credential.cert(serviceAccount),
    });
    console.log("Firebase Admin initialized with FIREBASE_SERVICE_ACCOUNT.");
  } catch (err) {
    console.error("Failed to parse FIREBASE_SERVICE_ACCOUNT JSON:", err);
    admin.initializeApp();
  }
} else {
  admin.initializeApp();
  console.log("Firebase Admin initialized with default credentials.");
}

const db = admin.firestore();
const app = express();
app.use(cors({ origin: true }));
app.use(express.json());

// System prompt defining the expert pedagogical persona for Civil Engineering students
const CIVIL_ENGINEERING_TUTOR_SYSTEM_INSTRUCTION = `
You are the CIVIL WORLD AI Tutor, a dedicated expert engineering educator designed specifically for Civil Engineering students.

SUBJECT COVERAGE:
- Engineering Mathematics, Mechanics, Structural Analysis, Strength of Materials.
- Reinforced Concrete Design (Eurocode 2, ACI 318, BS 8110, IS 456).
- Structural Steel Design (Eurocode 3, AISC 360, IS 800).
- Geotechnical Engineering, Soil Mechanics, Foundation Design.
- Fluid Mechanics, Open Channel Hydraulics, Hydrology, Water Supply & Sanitation.
- Transportation & Highway Engineering, Pavement Design.
- Surveying and Geomatics (Leveling, Theodolite Traverses, GPS, Total Station).
- Construction Technology, Building Materials (Concrete mix design, curing, masonry).
- Quantity Surveying, Bill of Quantities (BOQ), Cost Estimation.
- Environmental Engineering and Engineering Drawing / CAD.

TUTOR BEHAVIOURAL RULES:
1. Explain concepts in clear, student-friendly pedagogical language.
2. Provide systematic, step-by-step mathematical solutions.
3. Show all relevant governing formulas and explicitly define every single variable with units.
4. Always maintain strict SI metric units (kN, N, m, mm, kPa, MPa, kg/m³, m/s).
5. Highlight intermediate calculations and equilibrium checks (ΣFx = 0, ΣFy = 0, ΣM = 0).
6. Provide practical, real-world Civil Engineering context and construction site examples.
7. Explicitly state all structural or soil mechanics assumptions.
8. Clearly distinguish theoretical textbook academic derivations from actual building code / safety factor design recommendations.
9. Prompt the student for missing parameters when a structural problem or boundary condition is incomplete.
10. NEVER invent phantom building codes or state that an academic calculation is ready for field construction without certified Professional Engineer (PE) sign-off.
`;

/**
 * Middleware: Verifies Firebase Authentication token from Authorization header.
 * Attaches decoded user info to req.user.
 */
async function authenticateToken(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({
      error: "unauthenticated",
      message: "Authorization token required (Bearer <token>).",
    });
  }

  const token = authHeader.split(" ")[1];
  try {
    const decodedToken = await admin.auth().verifyIdToken(token);
    req.user = decodedToken;
    next();
  } catch (err) {
    console.error("Token verification failed:", err);
    return res.status(401).json({
      error: "unauthenticated",
      message: "Invalid or expired authentication token.",
    });
  }
}

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "CIVIL WORLD AI Backend",
    timestamp: new Date().toISOString(),
  });
});

/**
 * GET /api/entitlements
 * Retrieves or initializes student entitlement with 3/3 universal trial uses.
 */
app.get("/api/entitlements", authenticateToken, async (req, res) => {
  const uid = req.user.uid;
  const email = req.user.email || "";
  const isEmailVerified = req.user.email_verified === true;

  const userRef = db.collection("users").doc(uid);

  try {
    const result = await db.runTransaction(async (transaction) => {
      const doc = await transaction.get(userRef);

      if (!doc.exists) {
        const initialEntitlement = {
          uid: uid,
          email: email,
          accountPlan: "FREE",
          trialAllocated: 3,
          trialConsumed: 0,
          trialRemaining: 3,
          isTrialAwarded: true,
          subscriptionStatus: "NONE",
          subscriptionStartDate: null,
          subscriptionExpiryDate: null,
          isEmailVerified: isEmailVerified,
          createdAt: admin.firestore.FieldValue.serverTimestamp(),
          updatedAt: admin.firestore.FieldValue.serverTimestamp(),
        };

        transaction.set(userRef, initialEntitlement);
        return initialEntitlement;
      }

      const currentData = doc.data();

      // Check if subscription has expired
      if (
        currentData.subscriptionStatus === "ACTIVE" &&
        currentData.subscriptionExpiryDate
      ) {
        const expiry = currentData.subscriptionExpiryDate.toDate();
        if (new Date() > expiry) {
          transaction.update(userRef, {
            subscriptionStatus: "EXPIRED",
            accountPlan: "FREE",
            updatedAt: admin.firestore.FieldValue.serverTimestamp(),
          });
          currentData.subscriptionStatus = "EXPIRED";
          currentData.accountPlan = "FREE";
        }
      }

      return currentData;
    });

    res.json(result);
  } catch (err) {
    console.error("Error in getOrInitEntitlement:", err);
    res.status(500).json({ error: "internal", message: err.message });
  }
});

/**
 * POST /api/tutor/ask
 * Main Civil Engineering AI Tutor endpoint powered by Gemini.
 *
 * Verifies email verification status.
 * Enforces server-side trial deduction with idempotency protection.
 * Failed Gemini calls do NOT consume a trial use.
 */
app.post("/api/tutor/ask", authenticateToken, async (req, res) => {
  const uid = req.user.uid;
  const isEmailVerified = req.user.email_verified === true;

  if (!isEmailVerified) {
    return res.status(403).json({
      error: "failed-precondition",
      message: "Please verify your email before using the Civil Engineering AI Tutor.",
    });
  }

  const { prompt, conversationId, idempotencyKey } = req.body || {};

  if (!prompt || !prompt.trim()) {
    return res.status(400).json({
      error: "invalid-argument",
      message: "A prompt is required.",
    });
  }

  if (!idempotencyKey || !idempotencyKey.trim()) {
    return res.status(400).json({
      error: "invalid-argument",
      message: "An idempotencyKey is required.",
    });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({
      error: "failed-precondition",
      message: "Server environment GEMINI_API_KEY is not configured.",
    });
  }

  const userRef = db.collection("users").doc(uid);
  const historyRef = userRef.collection("ai_usage_history").doc(idempotencyKey);

  try {
    // 1. Idempotency Check
    const existingHistory = await historyRef.get();
    if (existingHistory.exists && existingHistory.data().aiResponse) {
      const data = existingHistory.data();
      return res.json({
        success: true,
        response: data.aiResponse,
        trialRemaining: data.remainingAfter,
        isPro: data.planUsed === "PRO",
        alreadyProcessed: true,
      });
    }

    // 2. Entitlement Check
    const userDoc = await userRef.get();
    let userData = userDoc.exists ? userDoc.data() : null;

    if (!userData) {
      // Auto-initialize if missing
      userData = {
        uid: uid,
        email: req.user.email || "",
        accountPlan: "FREE",
        trialAllocated: 3,
        trialConsumed: 0,
        trialRemaining: 3,
        subscriptionStatus: "NONE",
      };
      await userRef.set(userData);
    }

    const isPro =
      userData.accountPlan === "PRO" &&
      userData.subscriptionStatus === "ACTIVE" &&
      userData.subscriptionExpiryDate &&
      new Date() < userData.subscriptionExpiryDate.toDate();

    if (!isPro && (userData.trialRemaining || 0) <= 0) {
      return res.status(429).json({
        error: "resource-exhausted",
        message:
          "Your 3 universal AI trial uses have been exhausted. Please upgrade to CIVIL WORLD PRO to continue using the AI Tutor.",
      });
    }

    // 3. Query Gemini AI Engine via @google/generative-ai
    let generatedAiText = "";
    try {
      const genAI = new GoogleGenerativeAI(apiKey);
      const modelName = process.env.GEMINI_MODEL || "gemini-1.5-flash";

      const model = genAI.getGenerativeModel({
        model: modelName,
        systemInstruction: CIVIL_ENGINEERING_TUTOR_SYSTEM_INSTRUCTION,
        generationConfig: {
          temperature: 0.2,
          maxOutputTokens: 2048,
        },
      });

      const result = await model.generateContent(prompt.trim());
      const response = await result.response;
      generatedAiText = response.text();

      if (!generatedAiText) {
        throw new Error("Gemini returned an empty response.");
      }
    } catch (aiErr) {
      console.error("Gemini AI API Error:", aiErr);
      // NOTE: Failed Gemini requests are NOT deducted from the student's trial
      return res.status(502).json({
        error: "gemini-error",
        message: "Gemini AI Engine error: " + (aiErr.message || "Failed to generate answer") + ". No trial use was deducted.",
      });
    }

    // 4. Gemini Succeeded -> Atomic Entitlement Accounting
    let trialRemainingAfter = isPro
      ? userData.trialRemaining || 0
      : Math.max(0, (userData.trialRemaining || 0) - 1);

    await db.runTransaction(async (transaction) => {
      if (!isPro) {
        const freshUserDoc = await transaction.get(userRef);
        const freshData = freshUserDoc.data();
        const currentRemaining = freshData.trialRemaining || 0;

        if (currentRemaining <= 0) {
          throw new Error("Universal AI trial exhausted during processing.");
        }

        trialRemainingAfter = currentRemaining - 1;
        transaction.update(userRef, {
          trialRemaining: trialRemainingAfter,
          trialConsumed: (freshData.trialConsumed || 0) + 1,
          updatedAt: admin.firestore.FieldValue.serverTimestamp(),
        });
      }

      transaction.set(historyRef, {
        idempotencyKey: idempotencyKey,
        featureName: "AI Civil Engineering Tutor",
        featureDetails:
          prompt.length > 80 ? prompt.substring(0, 80) + "..." : prompt,
        planUsed: isPro ? "PRO" : "FREE_TRIAL",
        trialDeducted: !isPro,
        remainingAfter: trialRemainingAfter,
        aiResponse: generatedAiText,
        conversationId: conversationId || null,
        timestamp: admin.firestore.FieldValue.serverTimestamp(),
      });
    });

    return res.json({
      success: true,
      response: generatedAiText,
      isPro: isPro,
      trialRemaining: trialRemainingAfter,
      message: isPro
        ? "Response generated under CIVIL WORLD PRO membership."
        : `Universal AI trial task completed (${trialRemainingAfter}/3 uses remaining).`,
    });
  } catch (err) {
    console.error("Error processing tutor request:", err);
    return res.status(500).json({ error: "internal", message: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`CIVIL WORLD AI Backend listening on port ${PORT}`);
});

