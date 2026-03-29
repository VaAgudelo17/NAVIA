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
import { useFonts } from 'expo-font';
import { HomeScreen } from './src/screens/HomeScreen';
import { PreferencesProvider } from './src/context/PreferencesContext';
import { COLORS } from './src/constants/config';

export default function App() {
  // Cargar fuentes de íconos explícitamente para que funcionen en web
  const [fontsLoaded] = useFonts({
    Ionicons: require('@expo/vector-icons/build/vendor/react-native-vector-icons/Fonts/Ionicons.ttf'),
  });

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
    <PreferencesProvider>
      <SafeAreaView style={styles.container}>
        <StatusBar style="light" backgroundColor={COLORS.background} />
        <HomeScreen />
      </SafeAreaView>
    </PreferencesProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
});
