import path from "path";
import express from "express";

const app = express();

app.use(express.static("public"));

app.get("*", (req, res) => {
  res.sendFile(path.resolve("public", "index.html"));
});
