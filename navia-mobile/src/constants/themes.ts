/**
 * Temas visuales de NAVIA
 *
 * Diseñados para usuarios con diferentes niveles de visión residual:
 * - normal:          Tema oscuro estándar (por defecto)
 * - alto_contraste:  Negro/blanco máximo contraste
 * - amarillo_negro:  Amarillo sobre negro — óptimo para baja visión
 * - blanco_negro:    Fondo blanco — para quienes perciben luz
 * - azul_amarillo:   Azul/amarillo — daltonismo-friendly
 */

export type ThemeId =
  | 'normal'
  | 'alto_contraste'
  | 'amarillo_negro'
  | 'blanco_negro'
  | 'azul_amarillo';

export type FontSizeId = 'pequeno' | 'mediano' | 'grande';

export interface ThemeColors {
  background: string;
  surface: string;
  primary: string;
  primaryDark: string;
  secondary: string;
  text: string;
  textSecondary: string;
  error: string;
  success: string;
  warning: string;
  border: string;
}

interface ThemeDef {
  name: string;
  description: string;
  preview: [string, string, string]; // [bg, text, accent]
  colors: ThemeColors;
}

export const THEMES: Record<ThemeId, ThemeDef> = {
  normal: {
    name: 'Normal',
    description: 'Tema oscuro estándar',
    preview: ['#0a0e14', '#f8fafc', '#14b8a6'],
    colors: {
      background: '#0a0e14',
      surface: '#141b22',
      primary: '#14b8a6',
      primaryDark: '#0d9488',
      secondary: '#1e293b',
      text: '#f8fafc',
      textSecondary: '#94a3b8',
      error: '#ef4444',
      success: '#22c55e',
      warning: '#f59e0b',
      border: '#1e293b',
    },
  },
  alto_contraste: {
    name: 'Alto contraste',
    description: 'Negro sobre blanco',
    preview: ['#000000', '#ffffff', '#ffffff'],
    colors: {
      background: '#000000',
      surface: '#111111',
      primary: '#ffffff',
      primaryDark: '#cccccc',
      secondary: '#222222',
      text: '#ffffff',
      textSecondary: '#dddddd',
      error: '#ff4444',
      success: '#44ff44',
      warning: '#ffff44',
      border: '#555555',
    },
  },
  amarillo_negro: {
    name: 'Amarillo / Negro',
    description: 'Mejor para baja visión',
    preview: ['#000000', '#FFE600', '#FFE600'],
    colors: {
      background: '#000000',
      surface: '#0f0f00',
      primary: '#FFE600',
      primaryDark: '#CDB800',
      secondary: '#1a1a00',
      text: '#FFE600',
      textSecondary: '#CDB800',
      error: '#ff4444',
      success: '#66ff44',
      warning: '#FFE600',
      border: '#444400',
    },
  },
  blanco_negro: {
    name: 'Blanco / Negro',
    description: 'Para percepción de luz',
    preview: ['#ffffff', '#000000', '#000000'],
    colors: {
      background: '#ffffff',
      surface: '#f0f0f0',
      primary: '#000000',
      primaryDark: '#333333',
      secondary: '#e0e0e0',
      text: '#000000',
      textSecondary: '#555555',
      error: '#cc0000',
      success: '#006600',
      warning: '#996600',
      border: '#cccccc',
    },
  },
  azul_amarillo: {
    name: 'Azul / Amarillo',
    description: 'Daltonismo-friendly',
    preview: ['#001133', '#ffffff', '#FFE600'],
    colors: {
      background: '#001133',
      surface: '#002255',
      primary: '#FFE600',
      primaryDark: '#CDB800',
      secondary: '#003377',
      text: '#ffffff',
      textSecondary: '#aaccff',
      error: '#ff6666',
      success: '#66ff66',
      warning: '#FFE600',
      border: '#004499',
    },
  },
};

interface FontSizeDef {
  name: string;
  description: string;
  /** Factor multiplicador sobre la base de 16px */
  scale: number;
}

export const FONT_SIZES: Record<FontSizeId, FontSizeDef> = {
  pequeno: { name: 'Pequeño',  description: 'Texto compacto',   scale: 0.88 },
  mediano: { name: 'Mediano',  description: 'Tamaño estándar',  scale: 1.0  },
  grande:  { name: 'Grande',   description: 'Texto ampliado',   scale: 1.25 },
};

/** Escala un tamaño de fuente según la preferencia del usuario */
export function scaleFont(size: number, fontSizeId: FontSizeId): number {
  return Math.round(size * FONT_SIZES[fontSizeId].scale);
}
