const db = require("../config/db")

async function addVehicle(userId, type, brand, model, year, mileage) {
  await db.query(
    "INSERT INTO vehicles (user_id, type, brand, model, year, mileage) VALUES (?, ?, ?, ?, ?, ?)",
    [userId, type, brand, model, year, mileage]
  )
}

async function getVehiclesByUser(userId) {
  const [rows] = await db.query(
    "SELECT id, type, brand, model, year, mileage FROM vehicles WHERE user_id = ?",
    [userId]
  )
  return rows
}

module.exports = { addVehicle, getVehiclesByUser }
