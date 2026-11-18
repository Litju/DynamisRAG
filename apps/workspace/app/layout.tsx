import type { ReactNode } from "react";
import "./style.css";
export const metadata = {
  title: "Dynamis · Research workspace",
  description: "Evidence-grounded conversational research",
};
export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
