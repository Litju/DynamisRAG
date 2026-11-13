if (process.env.VERCEL_ENV !== "preview") {
  console.error(
    "RES-638 permits protected preview builds only; production is not authorized.",
  );
  process.exit(1);
}
