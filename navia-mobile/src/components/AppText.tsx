/**
 * AppText — Wrapper de Text que aplica la fuente seleccionada por el usuario.
 *
 * Reemplazar imports de Text de 'react-native' por este componente para que
 * la elección de tipografía en Configuración se aplique a TODA la app.
 * Si el style del usuario incluye fontFamily explícito, ese gana.
 */

import React from 'react';
import { Text as RNText, TextProps, StyleSheet } from 'react-native';
import { usePreferences } from '../context/PreferencesContext';

export function Text(props: TextProps) {
  const { fontFamily } = usePreferences();
  if (!fontFamily) {
    return <RNText {...props} />;
  }
  // El fontFamily va PRIMERO; cualquier style del usuario lo sobreescribe si lo necesita
  const flat = StyleSheet.flatten(props.style);
  const merged = { fontFamily, ...flat };
  return <RNText {...props} style={merged} />;
}

export default Text;
