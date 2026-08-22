"use client";

import Auralis from "@/components/ui/auralis";

export default function AuralisDemo() {
  return (
    <div className="relative w-full">
      <Auralis height="500px" />
      <div className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center px-6 text-center">
        <h1 className="text-4xl font-semibold tracking-tight text-white md:text-6xl">
          Auralis
        </h1>
        <p className="mt-4 max-w-md text-sm text-white/70 md:text-base">
          A WebGL ambient background with layered noise, glowing light, and film grain.
        </p>
      </div>
    </div>
  );
}
