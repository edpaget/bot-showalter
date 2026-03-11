// @refresh reset
import { createContext, type ReactNode, useCallback, useContext, useState } from "react";

interface PlayerDrawerState {
  isOpen: boolean;
  playerId: number | null;
  playerName: string | null;
  playerType: string | null;
  season: number;
  openPlayer: (playerId: number, playerName: string, playerType?: string) => void;
  closeDrawer: () => void;
}

const PlayerDrawerContext = createContext<PlayerDrawerState | null>(null);

export function PlayerDrawerProvider({ children, season = 2026 }: { children: ReactNode; season?: number }) {
  const [isOpen, setIsOpen] = useState(false);
  const [playerId, setPlayerId] = useState<number | null>(null);
  const [playerName, setPlayerName] = useState<string | null>(null);
  const [playerType, setPlayerType] = useState<string | null>(null);

  const openPlayer = useCallback((id: number, name: string, type?: string) => {
    setPlayerId(id);
    setPlayerName(name);
    setPlayerType(type ?? null);
    setIsOpen(true);
  }, []);

  const closeDrawer = useCallback(() => {
    setIsOpen(false);
  }, []);

  return (
    <PlayerDrawerContext.Provider
      value={{ isOpen, playerId, playerName, playerType, season, openPlayer, closeDrawer }}
    >
      {children}
    </PlayerDrawerContext.Provider>
  );
}

export function usePlayerDrawer(): PlayerDrawerState {
  const ctx = useContext(PlayerDrawerContext);
  if (!ctx) {
    throw new Error("usePlayerDrawer must be used within PlayerDrawerProvider");
  }
  return ctx;
}
