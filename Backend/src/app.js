const express = require("express")
const cors = require("cors")

const authRoutes = require("./routes/auth.routes")
const vehicleRoutes = require("./routes/vehicle.routes")

const app = express()

app.use(cors())
app.use(express.json())

app.use("/api/auth", authRoutes)
app.use("/api/vehicles", vehicleRoutes)

module.exports = app
