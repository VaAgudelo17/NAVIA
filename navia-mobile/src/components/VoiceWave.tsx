/**
 * VoiceWave — Animación de onda de voz para NAVIA
 *
 * Las barras SIEMPRE oscilan: con amplitud reducida en reposo,
 * con amplitud amplia y opacidad completa cuando el TTS está hablando.
 */

import React, { useEffect, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import { ttsManager } from '../services/ttsManager';

interface Props {
  color: string;
}

export function VoiceWave({ color }: Props) {
  const [tick, setTick] = useState(0);
  const [speaking, setSpeaking] = useState(false);

  useEffect(() => {
    let lastSpokeAt = 0;
    const interval = setInterval(() => {
      const now = Date.now();
      const isNow = ttsManager.isSpeaking();
      if (isNow) lastSpokeAt = now;
      setSpeaking(isNow || (now - lastSpokeAt) < 500);
      setTick((t) => (t + 1) % 1000);
    }, 80);
    return () => clearInterval(interval);
  }, []);

  const heights = [0, 1, 2, 3, 4].map((i) => {
    const phase = tick * 0.55 + i * 0.85;
    const sin = (Math.sin(phase) + 1) / 2;
    const amplitude = speaking ? 30 : 8;
    const base = speaking ? 10 : 12;
    return base + sin * amplitude;
  });

  return (
    <View
      style={styles.container}
      accessibilityElementsHidden
      importantForAccessibility="no-hide-descendants"
    >
      {heights.map((h, i) => (
        <View
          key={i}
          style={{
            width: 6,
            height: h,
            backgroundColor: color,
            opacity: speaking ? 1 : 0.5,
            borderRadius: 3,
            marginHorizontal: 4,
          }}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    height: 48,
    marginVertical: 12,
  },
});
