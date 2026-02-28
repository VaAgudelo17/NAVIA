"use client"

import { NaviaApp } from "@/components/navia/navia-app"
import { PreferencesProvider } from "@/context/PreferencesContext"

export default function Page() {
  return (
    <PreferencesProvider>
      <main>
        <NaviaApp />
      </main>
    </PreferencesProvider>
  )
}
