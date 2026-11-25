import { randomBytes, scryptSync } from "node:crypto";
import { readFileSync } from "node:fs";
// Read a password from stdin so it never enters shell history or process arguments.
const password = readFileSync(0, "utf8").trimEnd();
if (password.length < 16) throw new Error("Use at least 16 characters");
const salt = randomBytes(16).toString("hex");
process.stdout.write(`${salt}:${scryptSync(password, salt, 64).toString("hex")}\n`);
