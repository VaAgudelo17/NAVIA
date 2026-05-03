/**
 * Componente de botón accesible para NAVIA
 *
 * Lee los colores del tema activo vía usePreferences, así cualquier botón
 * de la app cambia de color cuando el usuario cambia de tema.
 */

import React from 'react';
import {
  TouchableOpacity,
  StyleSheet,
  ViewStyle,
  TextStyle,
  ActivityIndicator,
} from 'react-native';
import { Text } from './AppText';
import { usePreferences } from '../context/PreferencesContext';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'outline';
  size?: 'small' | 'medium' | 'large' | 'xl';
  disabled?: boolean;
  loading?: boolean;
  icon?: React.ReactNode;
  style?: ViewStyle;
  textStyle?: TextStyle;
  accessibilityLabel?: string;
}

export function Button({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  icon,
  style,
  textStyle,
  accessibilityLabel,
}: ButtonProps) {
  const { theme } = usePreferences();

  const variantStyle: ViewStyle =
    variant === 'primary'
      ? { backgroundColor: theme.primary }
      : variant === 'secondary'
      ? { backgroundColor: theme.secondary }
      : { backgroundColor: 'transparent', borderWidth: 2, borderColor: theme.primary };

  const variantTextColor =
    variant === 'primary' ? theme.background
    : variant === 'secondary' ? theme.text
    : theme.primary;

  const buttonStyles = [
    styles.button,
    variantStyle,
    styles[size],
    disabled && styles.disabled,
    style,
  ];

  const textStyles = [
    styles.text,
    { color: disabled ? theme.textSecondary : variantTextColor },
    styles[`${size}Text`],
    textStyle,
  ];

  return (
    <TouchableOpacity
      style={buttonStyles}
      onPress={onPress}
      disabled={disabled || loading}
      accessibilityLabel={accessibilityLabel || title}
      accessibilityRole="button"
      accessibilityState={{ disabled: disabled || loading }}
    >
      {loading ? (
        <ActivityIndicator color={variant === 'primary' ? theme.background : theme.primary} />
      ) : (
        <>
          {icon}
          <Text style={textStyles}>{title}</Text>
        </>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 12,
    gap: 8,
  },
  // Sizes
  small: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  medium: {
    paddingHorizontal: 24,
    paddingVertical: 12,
  },
  large: {
    paddingHorizontal: 32,
    paddingVertical: 16,
  },
  xl: {
    paddingHorizontal: 40,
    paddingVertical: 20,
  },
  text: {
    fontWeight: '600',
  },
  smallText: {
    fontSize: 14,
  },
  mediumText: {
    fontSize: 16,
  },
  largeText: {
    fontSize: 18,
  },
  xlText: {
    fontSize: 20,
  },
  disabled: {
    opacity: 0.5,
  },
});
