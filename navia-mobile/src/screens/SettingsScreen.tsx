/**
 * Pantalla de Configuración Visual — NAVIA
 *
 * Permite al usuario elegir:
 * - Tema de color (para visión residual / daltonismo)
 * - Tamaño de letra
 *
 * Aparece como modal animado desde abajo.
 * Cada opción tiene feedback TTS.
 */

import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Modal,
  Animated,
  Dimensions,
  StatusBar,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { usePreferences } from '../context/PreferencesContext';
import { THEMES, FONT_SIZES, ThemeId, FontSizeId } from '../constants/themes';
import { ttsManager, TtsPriority } from '../services/ttsManager';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');

interface SettingsScreenProps {
  visible: boolean;
  onClose: () => void;
}

const THEME_ORDER: ThemeId[] = [
  'normal',
  'alto_contraste',
  'amarillo_negro',
  'blanco_negro',
  'azul_amarillo',
];

const FONT_ORDER: FontSizeId[] = ['pequeno', 'mediano', 'grande'];

export function SettingsScreen({ visible, onClose }: SettingsScreenProps) {
  const { theme, fontScale, themeId, fontSizeId, setThemeId, setFontSizeId } = usePreferences();
  const slideAnim = useRef(new Animated.Value(SCREEN_HEIGHT)).current;

  useEffect(() => {
    if (visible) {
      Animated.spring(slideAnim, {
        toValue: 0,
        useNativeDriver: true,
        tension: 65,
        friction: 11,
      }).start();
      ttsManager.speak('Abriste la configuración visual. Puedes elegir el tema de color que mejor te funcione, o ajustar el tamaño de la letra.', TtsPriority.HIGH);
    } else {
      Animated.timing(slideAnim, {
        toValue: SCREEN_HEIGHT,
        duration: 260,
        useNativeDriver: true,
      }).start();
    }
  }, [visible]);

  const handleThemeSelect = (id: ThemeId) => {
    setThemeId(id);
    const t = THEMES[id];
    const nombre = t.ttsName ?? t.name;
    ttsManager.stop();
    ttsManager.speak(`Perfecto, elegiste el tema ${nombre}. ${t.description}.`, TtsPriority.INTERRUPT);
  };

  const handleFontSelect = (id: FontSizeId) => {
    setFontSizeId(id);
    const f = FONT_SIZES[id];
    ttsManager.stop();
    ttsManager.speak(`Listo, cambiaste el tamaño de letra a ${f.name}. ${f.description}.`, TtsPriority.INTERRUPT);
  };

  const handleClose = () => {
    ttsManager.stop();
    ttsManager.speak('Configuración guardada. De vuelta al inicio.', TtsPriority.INTERRUPT);
    onClose();
  };

  const fs = (size: number) => Math.round(size * fontScale);

  return (
    <Modal
      visible={visible}
      transparent
      animationType="none"
      statusBarTranslucent
      onRequestClose={handleClose}
    >
      {/* Fondo semitransparente */}
      <TouchableOpacity
        style={styles.backdrop}
        activeOpacity={1}
        onPress={handleClose}
        accessibilityLabel="Cerrar configuración"
      />

      <Animated.View
        style={[
          styles.sheet,
          { backgroundColor: theme.surface, transform: [{ translateY: slideAnim }] },
        ]}
      >
        {/* Handle visual */}
        <View style={styles.handleContainer}>
          <View style={[styles.handle, { backgroundColor: theme.border }]} />
        </View>

        {/* Header */}
        <View style={[styles.header, { borderBottomColor: theme.border }]}>
          <View style={styles.headerLeft}>
            <Ionicons name="settings" size={24} color={theme.primary} />
            <Text style={[styles.headerTitle, { color: theme.text, fontSize: fs(20) }]}>
              Configuración Visual
            </Text>
          </View>
          <TouchableOpacity
            onPress={handleClose}
            style={[styles.closeBtn, { backgroundColor: theme.secondary }]}
            accessibilityLabel="Cerrar"
            accessibilityRole="button"
          >
            <Ionicons name="close" size={20} color={theme.text} />
          </TouchableOpacity>
        </View>

        <ScrollView
          style={styles.scroll}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* ---- Sección Temas ---- */}
          <Text style={[styles.sectionTitle, { color: theme.textSecondary, fontSize: fs(12) }]}>
            TEMA DE COLOR
          </Text>
          <Text style={[styles.sectionDesc, { color: theme.textSecondary, fontSize: fs(13) }]}>
            Diseñados para diferentes niveles de visión residual
          </Text>

          {THEME_ORDER.map((id) => {
            const t = THEMES[id];
            const isActive = themeId === id;
            const [bg, fg, accent] = t.preview;
            return (
              <TouchableOpacity
                key={id}
                style={[
                  styles.themeCard,
                  {
                    backgroundColor: isActive ? theme.primary + '22' : theme.secondary,
                    borderColor: isActive ? theme.primary : theme.border,
                    borderWidth: isActive ? 2 : 1,
                  },
                ]}
                onPress={() => handleThemeSelect(id)}
                accessibilityRole="radio"
                accessibilityState={{ selected: isActive }}
                accessibilityLabel={`Tema ${t.name}. ${t.description}`}
              >
                {/* Preview de colores */}
                <View style={[styles.previewBox, { backgroundColor: bg, borderColor: theme.border }]}>
                  <View style={[styles.previewBar, { backgroundColor: accent }]} />
                  <View style={styles.previewLines}>
                    <View style={[styles.previewLine, { backgroundColor: fg, width: '70%' }]} />
                    <View style={[styles.previewLine, { backgroundColor: fg, width: '50%', opacity: 0.6 }]} />
                  </View>
                </View>

                {/* Info */}
                <View style={styles.themeInfo}>
                  <Text style={[styles.themeName, { color: theme.text, fontSize: fs(15) }]}>
                    {t.name}
                  </Text>
                  <Text style={[styles.themeDesc, { color: theme.textSecondary, fontSize: fs(12) }]}>
                    {t.description}
                  </Text>
                </View>

                {/* Check activo */}
                {isActive && (
                  <View style={[styles.activeCheck, { backgroundColor: theme.primary }]}>
                    <Ionicons name="checkmark" size={16} color={theme.background} />
                  </View>
                )}
              </TouchableOpacity>
            );
          })}

          {/* ---- Sección Tamaño de letra ---- */}
          <Text style={[styles.sectionTitle, { color: theme.textSecondary, fontSize: fs(12), marginTop: 28 }]}>
            TAMAÑO DE LETRA
          </Text>

          <View style={styles.fontRow}>
            {FONT_ORDER.map((id) => {
              const f = FONT_SIZES[id];
              const isActive = fontSizeId === id;
              return (
                <TouchableOpacity
                  key={id}
                  style={[
                    styles.fontCard,
                    {
                      backgroundColor: isActive ? theme.primary + '22' : theme.secondary,
                      borderColor: isActive ? theme.primary : theme.border,
                      borderWidth: isActive ? 2 : 1,
                    },
                  ]}
                  onPress={() => handleFontSelect(id)}
                  accessibilityRole="radio"
                  accessibilityState={{ selected: isActive }}
                  accessibilityLabel={`Letra ${f.name}. ${f.description}`}
                >
                  <Text
                    style={[
                      styles.fontSample,
                      {
                        color: isActive ? theme.primary : theme.text,
                        fontSize: Math.round(14 * f.scale),
                      },
                    ]}
                  >
                    Aa
                  </Text>
                  <Text style={[styles.fontName, { color: theme.text, fontSize: fs(13) }]}>
                    {f.name}
                  </Text>
                  <Text style={[styles.fontDesc, { color: theme.textSecondary, fontSize: fs(11) }]}>
                    {f.description}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>

          {/* Preview del texto actual */}
          <View style={[styles.previewTextBox, { backgroundColor: theme.secondary, borderColor: theme.border }]}>
            <Text style={[styles.previewLabel, { color: theme.textSecondary, fontSize: fs(11) }]}>
              VISTA PREVIA
            </Text>
            <Text style={[styles.previewText, { color: theme.text, fontSize: fs(16) }]}>
              NAVIA · Asistente Visual
            </Text>
            <Text style={[styles.previewSubtext, { color: theme.textSecondary, fontSize: fs(13) }]}>
              Navegación asistida con inteligencia artificial
            </Text>
          </View>

          <View style={{ height: 32 }} />
        </ScrollView>
      </Animated.View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.6)',
  },
  sheet: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    maxHeight: SCREEN_HEIGHT * 0.92,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    overflow: 'hidden',
  },
  handleContainer: {
    alignItems: 'center',
    paddingTop: 12,
    paddingBottom: 4,
  },
  handle: {
    width: 40,
    height: 4,
    borderRadius: 2,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  headerTitle: {
    fontWeight: '700',
  },
  closeBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
  },
  sectionTitle: {
    fontWeight: '700',
    letterSpacing: 1.2,
    marginBottom: 4,
  },
  sectionDesc: {
    marginBottom: 16,
  },
  // Tarjeta de tema
  themeCard: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    gap: 14,
  },
  previewBox: {
    width: 64,
    height: 48,
    borderRadius: 10,
    overflow: 'hidden',
    borderWidth: 1,
  },
  previewBar: {
    height: 14,
    width: '100%',
  },
  previewLines: {
    flex: 1,
    padding: 6,
    gap: 4,
  },
  previewLine: {
    height: 3,
    borderRadius: 2,
  },
  themeInfo: {
    flex: 1,
    gap: 2,
  },
  themeName: {
    fontWeight: '600',
  },
  themeDesc: {
    lineHeight: 16,
  },
  activeCheck: {
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  // Tarjetas de tamaño
  fontRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 20,
  },
  fontCard: {
    flex: 1,
    alignItems: 'center',
    borderRadius: 16,
    paddingVertical: 16,
    paddingHorizontal: 8,
    gap: 4,
  },
  fontSample: {
    fontWeight: '700',
  },
  fontName: {
    fontWeight: '600',
  },
  fontDesc: {
    textAlign: 'center',
    lineHeight: 14,
  },
  // Preview de texto
  previewTextBox: {
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    gap: 6,
  },
  previewLabel: {
    fontWeight: '700',
    letterSpacing: 1,
  },
  previewText: {
    fontWeight: '700',
  },
  previewSubtext: {
    lineHeight: 18,
  },
});
