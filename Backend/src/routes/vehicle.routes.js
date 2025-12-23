const express = require("express")
const auth = require("../middleware/auth.middleware")
const { createVehicle, getVehicles } = require("../controllers/vehicle.controller")

const router = express.Router()

router.post("/", auth, createVehicle)
router.get("/", auth, getVehicles)

module.exports = router
