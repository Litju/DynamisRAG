FROM node:24-bookworm-slim@sha256:dc26ae92c9fb15cf6b1d63e20f0944b2efe0f115b20e00be48fb23fceb061c54
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1 EVE_TELEMETRY_DISABLED=1 NEXT_PUBLIC_RESEARCH_MODE=owner-local
RUN npx --no-install eve build && npm run build && chown -R node:node /app/.eve /app/.next
USER node
