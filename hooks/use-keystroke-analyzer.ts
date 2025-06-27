"use client"

import type React from "react"
import { useState, useCallback } from "react"

interface KeystrokeEvent {
  key: string
  type: "keydown" | "keyup"
  timestamp: number
}

interface ExtractedFeatures {
  holdTimes: number[]
  ddTimes: number[]
  udTimes: number[]
  typingSpeed: number
  flightTime: number
  errorRate: number
  pressPressure: number
  features: number[]
}

export function useKeystrokeAnalyzer() {
  const [keystrokeData, setKeystrokeData] = useState<KeystrokeEvent[]>([])
  const [isCapturing, setIsCapturing] = useState(false)

  const captureKeystrokes = useCallback((event: React.KeyboardEvent, type: "keydown" | "keyup") => {
    const keystroke: KeystrokeEvent = {
      key: event.key,
      type,
      timestamp: performance.now(), // High precision timing
    }

    setKeystrokeData((prev) => [...prev, keystroke])
  }, [])

  const extractFeatures = useCallback((data: KeystrokeEvent[]): ExtractedFeatures => {
    // Split events by type for easier processing
    const keydowns = data.filter((k) => k.type === "keydown")
    const keyups = data.filter((k) => k.type === "keyup")

    // Create matched pairs of keydown events with their timestamps
    const matchedKeys = keydowns.map((k) => [k.key, k.timestamp] as [string, number])

    const holdTimes: number[] = []
    const ddTimes: number[] = [] // dwell times (down-down)
    const udTimes: number[] = [] // flight times (up-down)

    // Calculate hold times - time between press and release
    matchedKeys.forEach(([key, downTs]) => {
      const upEvent = keyups.find((u) => u.key === key && u.timestamp > downTs)
      if (upEvent) {
        holdTimes.push(upEvent.timestamp - downTs)
      }
    })

    // Calculate dwell times - time between consecutive key presses
    for (let i = 0; i < matchedKeys.length - 1; i++) {
      const press1 = matchedKeys[i][1]
      const press2 = matchedKeys[i + 1][1]
      ddTimes.push(press2 - press1)
    }

    // Calculate flight times - time from key release to next key press
    for (let i = 0; i < matchedKeys.length - 1; i++) {
      const currentKey = matchedKeys[i][0]
      const currentDown = matchedKeys[i][1]
      const nextDown = matchedKeys[i + 1][1]

      const currentUp = keyups.find((u) => u.key === currentKey && u.timestamp > currentDown)
      if (currentUp) {
        udTimes.push(nextDown - currentUp.timestamp)
      } else {
        // Fallback if we can't find the up event
        udTimes.push(nextDown - currentDown)
      }
    }

    // Calculate derived metrics
    const totalTime =
      Math.max(
        holdTimes.reduce((sum, t) => sum + t, 0),
        ddTimes.reduce((sum, t) => sum + t, 0),
        udTimes.reduce((sum, t) => sum + t, 0),
      ) || 0.001 // Avoid division by zero

    const typingSpeed = matchedKeys.length / (totalTime / 1000) // keys per second
    const flightTime = udTimes.length > 0 ? udTimes.reduce((a, b) => a + b, 0) / udTimes.length : 0
    const errorRate = data.filter((k) => k.key === "Backspace").length

    // Calculate press pressure variance (how consistent the hold times are)
    const meanHoldTime = holdTimes.length > 0 ? holdTimes.reduce((a, b) => a + b, 0) / holdTimes.length : 0
    const pressPressure =
      holdTimes.length > 0
        ? Math.sqrt(holdTimes.reduce((sum, t) => sum + Math.pow(t - meanHoldTime, 2), 0) / holdTimes.length)
        : 0

    // Build feature vector to match the Python implementation
    const PASSWORD_LENGTH = 11
    const featureVector = [
      ...holdTimes.slice(0, PASSWORD_LENGTH),
      ...ddTimes.slice(0, PASSWORD_LENGTH - 1),
      ...udTimes.slice(0, PASSWORD_LENGTH - 1),
      typingSpeed,
      flightTime,
      errorRate,
      pressPressure,
    ]

    // Pad with zeros if we don't have enough features
    while (featureVector.length < PASSWORD_LENGTH * 3 + 1) {
      featureVector.push(0)
    }

    return {
      holdTimes,
      ddTimes,
      udTimes,
      typingSpeed,
      flightTime,
      errorRate,
      pressPressure,
      features: featureVector,
    }
  }, [])

  const trainModel = useCallback(
    async (username: string, features: ExtractedFeatures, sampleCount: number, privacyMode: boolean) => {
      try {
        const response = await fetch("/api/train-model", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            username,
            features: features.features,
            holdTimes: features.holdTimes,
            ddTimes: features.ddTimes,
            udTimes: features.udTimes,
            additionalFeatures: {
              typingSpeed: features.typingSpeed,
              flightTime: features.flightTime,
              errorRate: features.errorRate,
              pressPressure: features.pressPressure,
            },
            sampleCount,
            privacyMode,
            rawData: privacyMode ? null : keystrokeData, // Only include raw data if not in privacy mode
          }),
        })

        const result = await response.json()
        return result.success
      } catch (error) {
        console.error("Training failed:", error)
        return false
      }
    },
    [keystrokeData],
  )

  const authenticate = useCallback(async (username: string, features: ExtractedFeatures, password: string) => {
    try {
      console.log("Authenticating user:", username)
      console.log("Features:", features.features.slice(0, 5), "...") // Log first 5 features for debugging

      const response = await fetch("/api/authenticate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username,
          features: features.features,
          holdTimes: features.holdTimes,
          ddTimes: features.ddTimes,
          udTimes: features.udTimes,
          typingSpeed: features.typingSpeed,
          flightTime: features.flightTime,
          errorRate: features.errorRate,
          pressPressure: features.pressPressure,
          password,
        }),
      })

      const result = await response.json()
      console.log("Authentication result:", result)
      return result
    } catch (error) {
      console.error("Authentication failed:", error)
      return { success: false, authenticated: false, mse: 0, deviations: [] }
    }
  }, [])

  const resetCapture = useCallback(() => {
    setKeystrokeData([])
  }, [])

  return {
    captureKeystrokes,
    extractFeatures,
    trainModel,
    authenticate,
    resetCapture,
    isCapturing,
    keystrokeData,
  }
}