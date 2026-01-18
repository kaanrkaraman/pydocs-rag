# Python Docs RAG Frontend

Modern, responsive web interface for the Python Documentation RAG system built with React, TypeScript, Vite, and Tailwind 4.

## Tech Stack

- **React 19** - UI library with latest features
- **TypeScript 5.9** - Type-safe development
- **Vite 7** - Lightning-fast build tool and dev server
- **Tailwind 4** - CSS-first styling framework
- **Framer Motion** - Production-ready animation library
- **Biome** - Fast linter and formatter
- **Bun** - JavaScript runtime and package manager

## Project Structure

```
frontend/
├── src/
│   ├── components/     # React components
│   ├── hooks/          # Custom React hooks
│   ├── lib/            # Utilities and API client
│   ├── types/          # TypeScript type definitions
│   ├── styles/         # Additional styles
│   ├── assets/         # Images, fonts, etc.
│   ├── App.tsx         # Root component
│   ├── main.tsx        # Application entry point
│   └── index.css       # Tailwind imports and global styles
├── public/             # Static assets
└── vite.config.ts      # Vite configuration
```

## Getting Started

### Prerequisites

- Bun 1.0+

### Installation

```bash
bun install
```

### Development

Start the dev server on http://localhost:5173:

```bash
bun run dev
```

The dev server proxies API requests to `http://localhost:8000` (FastAPI backend).

### Building

Build for production:

```bash
bun run build
```

Preview production build:

```bash
bun run preview
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
VITE_API_URL=http://localhost:8000
VITE_APP_NAME="Python Docs RAG"
VITE_APP_VERSION=0.1.0
```

## Code Quality

### Linting

```bash
bun run lint
```

### Formatting

```bash
bun run format
```

### Check (Lint + Format)

```bash
bun run check
```

### Type Checking

TypeScript is checked during build. For manual checking:

```bash
bun run typecheck
```

### Design Principles

This project follows the frontend design principles outlined in `../CLAUDE.md`:

- Distinctive, non-generic aesthetics
- Creative typography choices
- Cohesive color schemes with dominant + accent approach
- Purposeful animations and micro-interactions
- Atmospheric backgrounds with depth
- Context-specific, production-ready designs

## API Integration

The API client (`src/lib/api.ts`) provides type-safe methods for:

- `query(query: string)` - Submit queries to RAG system
- `health()` - Check API health status
- `indexInfo()` - Get index information
- `stats()` - Retrieve usage statistics

## Path Aliases

Import using `@/` prefix for cleaner imports:

```typescript
import { cn } from "@/lib/utils";
import type { QueryResponse } from "@/types/api";
```

## Scripts Reference

- `bun run dev` - Start development server
- `bun run build` - Build for production
- `bun run preview` - Preview production build
- `bun run lint` - Lint TypeScript/TSX files
- `bun run format` - Format TypeScript/TSX files
- `bun run check` - Lint and format in one command
- `bun run typecheck` - Type check without emitting files
