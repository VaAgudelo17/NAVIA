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
import { AtkinsonHyperlegible_400Regular, AtkinsonHyperlegible_700Bold } from '@expo-google-fonts/atkinson-hyperlegible';
import { Roboto_400Regular, Roboto_700Bold } from '@expo-google-fonts/roboto';
import { HomeScreen } from './src/screens/HomeScreen';
import { PreferencesProvider } from './src/context/PreferencesContext';
import { COLORS } from './src/constants/config';

export default function App() {
  // Cargar fuentes de íconos explícitamente para que funcionen en web,
  // junto con Atkinson Hyperlegible y Roboto para preferencias de accesibilidad
  const [fontsLoaded] = useFonts({
    Ionicons: require('@expo/vector-icons/build/vendor/react-native-vector-icons/Fonts/Ionicons.ttf'),
    AtkinsonHyperlegible_400Regular,
    AtkinsonHyperlegible_700Bold,
    Roboto_400Regular,
    Roboto_700Bold,
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

  // No renderizar hasta que las fuentes (incluida Atkinson y Roboto) estén listas
  if (!fontsLoaded) return null;

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
