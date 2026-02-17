/**
 * NAVIA - Aplicación de Asistencia Visual con IA
 *
 * Proyecto de tesis universitaria
 * Universidad Simón Bolívar
 */

import React, { useEffect } from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { Audio } from 'expo-av';
import { HomeScreen } from './src/screens/HomeScreen';
import { COLORS } from './src/constants/config';

export default function App() {
  useEffect(() => {
    // Configurar audio session para que TTS funcione con el switch de silencio en iOS
    // y baje el volumen de otras apps en Android
    Audio.setAudioModeAsync({
      playsInSilentModeIOS: true,
      staysActiveInBackground: false,
      shouldDuckAndroid: true,
    });
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" backgroundColor={COLORS.background} />
      <HomeScreen />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
});
