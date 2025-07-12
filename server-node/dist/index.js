"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const multer_1 = __importDefault(require("multer"));
const cors_1 = __importDefault(require("cors"));
const fs_1 = __importDefault(require("fs"));
const form_data_1 = __importDefault(require("form-data"));
const axios_1 = __importDefault(require("axios"));
const util_1 = require("util");
const stream_1 = require("stream");
require('dotenv').config();
const app = (0, express_1.default)();
const PORT = process.env.PORT || 5000;
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || "http://127.0.0.1:8000";
app.use((0, cors_1.default)());
app.use(express_1.default.json());
// Configure multer for file uploads
const storage = multer_1.default.diskStorage({
    destination: "uploads/",
    filename: (_req, file, cb) => {
        cb(null, Date.now() + "-" + file.originalname);
    },
});
const upload = (0, multer_1.default)({
    storage,
    limits: {
        fileSize: 50 * 1024 * 1024, // 50MB limit
    },
    fileFilter: (_req, file, cb) => {
        const allowedTypes = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg', 'image/bmp', 'image/tiff'];
        if (allowedTypes.includes(file.mimetype)) {
            cb(null, true);
        }
        else {
            cb(new Error('Invalid file type. Only PDF and image files are allowed.'));
        }
    }
});
// Ensure upload directory exists
if (!fs_1.default.existsSync("uploads")) {
    fs_1.default.mkdirSync("uploads", { recursive: true });
}
// Health check endpoint
app.get("/health", async (req, res) => {
    try {
        // Check if Python service is running
        const pythonHealth = await axios_1.default.get(`${PYTHON_SERVICE_URL}/health`, {
            timeout: 5000
        });
        res.json({
            status: "healthy",
            nodejs: "running",
            python_service: pythonHealth.data.status,
            timestamp: new Date().toISOString()
        });
    }
    catch (error) {
        res.status(503).json({
            status: "degraded",
            nodejs: "running",
            python_service: "unavailable",
            error: error instanceof Error ? error.message : "Unknown error",
            timestamp: new Date().toISOString()
        });
    }
});
// Main conversion endpoint
app.post("/convert", upload.single("file"), async (req, res) => {
    const startTime = Date.now();
    try {
        const file = req.file;
        if (!file) {
            res.status(400).json({
                error: "No file uploaded",
                message: "Please select a file to convert"
            });
            return;
        }
        console.log(`Processing file: ${file.originalname}, Size: ${file.size} bytes`);
        // Validate file size
        if (file.size > 50 * 1024 * 1024) { // 50MB
            res.status(400).json({
                error: "File too large",
                message: "File size must be less than 50MB"
            });
            return;
        }
        // Create form data for Python service
        const formData = new form_data_1.default();
        const fileStream = fs_1.default.createReadStream(file.path);
        formData.append('file', fileStream, {
            filename: file.originalname,
            contentType: file.mimetype
        });
        // Call Python conversion service
        const response = await axios_1.default.post(`${PYTHON_SERVICE_URL}/convert`, formData, {
            headers: {
                ...formData.getHeaders(),
            },
            responseType: 'stream',
            timeout: 300000, // 5 minutes timeout
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });
        // Set response headers
        const filename = file.originalname.replace(/\.[^/.]+$/, "") + ".docx";
        res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
        res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
        // Stream the response back to client
        const streamPipeline = (0, util_1.promisify)(stream_1.pipeline);
        await streamPipeline(response.data, res);
        const processingTime = Date.now() - startTime;
        console.log(`File converted successfully in ${processingTime}ms: ${file.originalname}`);
    }
    catch (error) {
        console.error("Conversion error:", error);
        if (!res.headersSent) {
            if (axios_1.default.isAxiosError(error)) {
                if (error.code === 'ECONNREFUSED') {
                    res.status(503).json({
                        error: "Service unavailable",
                        message: "PDF conversion service is currently unavailable. Please try again later."
                    });
                }
                else if (error.response?.status === 400) {
                    res.status(400).json({
                        error: "Invalid file",
                        message: "The uploaded file could not be processed. Please ensure it's a valid PDF or image file."
                    });
                }
                else if (error.code === 'ECONNABORTED') {
                    res.status(408).json({
                        error: "Request timeout",
                        message: "File conversion took too long. Please try with a smaller file."
                    });
                }
                else {
                    res.status(500).json({
                        error: "Conversion failed",
                        message: "An error occurred during file conversion. Please try again."
                    });
                }
            }
            else {
                res.status(500).json({
                    error: "Internal server error",
                    message: "An unexpected error occurred. Please try again."
                });
            }
        }
    }
    finally {
        // Clean up uploaded file
        if (req.file?.path) {
            try {
                fs_1.default.unlinkSync(req.file.path);
            }
            catch (cleanupError) {
                console.error("File cleanup error:", cleanupError);
            }
        }
    }
});
// Get conversion status endpoint
app.get("/status", (req, res) => {
    res.json({
        service: "PDF to Word Converter",
        version: "1.0.0",
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        timestamp: new Date().toISOString()
    });
});
// Error handling middleware
app.use((error, req, res, next) => {
    console.error("Unhandled error:", error);
    if (error instanceof multer_1.default.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            res.status(400).json({
                error: "File too large",
                message: "File size must be less than 50MB"
            });
        }
        else {
            res.status(400).json({
                error: "Upload error",
                message: error.message
            });
        }
    }
    else {
        res.status(500).json({
            error: "Internal server error",
            message: "An unexpected error occurred"
        });
    }
});
app.listen(PORT, () => {
    console.log(`🚀 Node.js API server running on port ${PORT}`);
    console.log(`📡 Python service expected at ${PYTHON_SERVICE_URL}`);
    console.log(`🔍 Health check: http://localhost:${PORT}/health`);
});
