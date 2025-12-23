const { addVehicle, getVehiclesByUser } = require("../models/vehicle.model")

async function createVehicle(req, res) {
  const { type, brand, model, year, mileage } = req.body
  await addVehicle(req.user.id, type, brand, model, year, mileage)
  res.json({ message: "Vehicle added successfully" })
}

async function getVehicles(req, res) {
  const vehicles = await getVehiclesByUser(req.user.id)
  res.json(vehicles)
}

module.exports = { createVehicle, getVehicles }
