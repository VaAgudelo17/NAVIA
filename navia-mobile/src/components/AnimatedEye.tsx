import React, { useEffect, useRef } from 'react';
import { Animated, Easing, View, StyleSheet } from 'react-native';
import Svg, { Path, Circle, Line } from 'react-native-svg';

interface AnimatedEyeProps {
  size?: number;
  color?: string;
}

export function AnimatedEye({ size = 48, color = '#14b8a6' }: AnimatedEyeProps) {
  const scaleY = useRef(new Animated.Value(1)).current;
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    const blink = () => {
      Animated.sequence([
        // Cerrar
        Animated.timing(scaleY, {
          toValue: 0.1,
          duration: 150,
          easing: Easing.in(Easing.ease),
          useNativeDriver: true,
        }),
        // Mantener cerrado
        Animated.delay(80),
        // Abrir
        Animated.timing(scaleY, {
          toValue: 1,
          duration: 200,
          easing: Easing.out(Easing.ease),
          useNativeDriver: true,
        }),
      ]).start(() => {
        if (!cancelled) {
          const delay = 3000 + Math.random() * 3000;
          timeoutRef.current = setTimeout(blink, delay);
        }
      });
    };

    const initialDelay = 2000 + Math.random() * 2000;
    timeoutRef.current = setTimeout(blink, initialDelay);

    return () => {
      cancelled = true;
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      scaleY.stopAnimation();
    };
  }, [scaleY]);

  return (
    <View style={[styles.container, { width: size, height: size }]}>
      {/* Capa 1: Pestañas (siempre visibles) */}
      <Svg
        width={size}
        height={size}
        viewBox="0 0 64 64"
        style={StyleSheet.absoluteFill}
      >
        <Line x1="22" y1="18" x2="20" y2="12" stroke={color} strokeWidth="2" strokeLinecap="round" />
        <Line x1="32" y1="16" x2="32" y2="9" stroke={color} strokeWidth="2" strokeLinecap="round" />
        <Line x1="42" y1="18" x2="44" y2="12" stroke={color} strokeWidth="2" strokeLinecap="round" />
      </Svg>

      {/* Capa 2: Cuerpo del ojo (se anima con scaleY) */}
      <Animated.View
        style={[
          StyleSheet.absoluteFill,
          { transform: [{ scaleY: scaleY }] },
        ]}
      >
        <Svg width={size} height={size} viewBox="0 0 64 64">
          {/* Contorno almendrado */}
          <Path
            d="M8 32 C8 32, 20 16, 32 16 C44 16, 56 32, 56 32 C56 32, 44 48, 32 48 C20 48, 8 32, 8 32 Z"
            stroke={color}
            strokeWidth="2"
            fill="none"
          />
          {/* Iris */}
          <Circle cx="32" cy="32" r="11" fill={color} />
          {/* Pupila */}
          <Circle cx="32" cy="32" r="5" fill="#1a1a2e" />
          {/* Brillo */}
          <Circle cx="28" cy="28" r="1.5" fill="rgba(255,255,255,0.7)" />
        </Svg>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
});
