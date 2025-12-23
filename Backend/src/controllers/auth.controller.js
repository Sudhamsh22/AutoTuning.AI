const bcrypt = require("bcryptjs")
const jwt = require("jsonwebtoken")
const { findUserByEmail, createUser } = require("../models/user.model")

async function signup(req, res) {
  const { full_name, email, password } = req.body

  const existingUser = await findUserByEmail(email)
  if (existingUser) return res.status(400).json({ message: "Email already exists" })

  const hash = await bcrypt.hash(password, 10)
  await createUser(full_name, email, hash)

  res.json({ message: "Account created successfully" })
}

async function login(req, res) {
  const { email, password } = req.body

  const user = await findUserByEmail(email)
  if (!user) return res.status(400).json({ message: "Invalid credentials" })

  const match = await bcrypt.compare(password, user.password_hash)
  if (!match) return res.status(400).json({ message: "Invalid credentials" })

  const token = jwt.sign(
    { id: user.id, email: user.email },
    process.env.JWT_SECRET,
    { expiresIn: "7d" }
  )

  res.json({
    token,
    user: {
      id: user.id,
      full_name: user.full_name,
      email: user.email
    }
  })
}

module.exports = { signup, login }
