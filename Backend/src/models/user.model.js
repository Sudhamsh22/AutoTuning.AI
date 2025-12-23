const db = require("../config/db")

async function findUserByEmail(email) {
  const [rows] = await db.query("SELECT * FROM users WHERE email = ?", [email])
  return rows[0]
}

async function createUser(fullName, email, passwordHash) {
  const [result] = await db.query(
    "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
    [fullName, email, passwordHash]
  )
  return result.insertId
}

module.exports = { findUserByEmail, createUser }
