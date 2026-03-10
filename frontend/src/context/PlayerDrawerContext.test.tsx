import { render, screen, act, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, afterEach } from "vitest";
import { PlayerDrawerProvider, usePlayerDrawer } from "./PlayerDrawerContext";

function TestConsumer() {
  const { isOpen, playerId, playerName, openPlayer, closeDrawer } = usePlayerDrawer();
  return (
    <div>
      <span data-testid="open">{String(isOpen)}</span>
      <span data-testid="id">{String(playerId)}</span>
      <span data-testid="name">{String(playerName)}</span>
      <button data-testid="open-btn" onClick={() => openPlayer(42, "Mike Trout")}>
        Open
      </button>
      <button data-testid="close-btn" onClick={closeDrawer}>
        Close
      </button>
    </div>
  );
}

describe("PlayerDrawerContext", () => {
  afterEach(cleanup);

  it("starts closed with null values", () => {
    render(
      <PlayerDrawerProvider>
        <TestConsumer />
      </PlayerDrawerProvider>,
    );
    expect(screen.getByTestId("open").textContent).toBe("false");
    expect(screen.getByTestId("id").textContent).toBe("null");
    expect(screen.getByTestId("name").textContent).toBe("null");
  });

  it("opens with player data", async () => {
    render(
      <PlayerDrawerProvider>
        <TestConsumer />
      </PlayerDrawerProvider>,
    );
    await act(() => userEvent.click(screen.getByTestId("open-btn")));
    expect(screen.getByTestId("open").textContent).toBe("true");
    expect(screen.getByTestId("id").textContent).toBe("42");
    expect(screen.getByTestId("name").textContent).toBe("Mike Trout");
  });

  it("closes drawer", async () => {
    render(
      <PlayerDrawerProvider>
        <TestConsumer />
      </PlayerDrawerProvider>,
    );
    await act(() => userEvent.click(screen.getByTestId("open-btn")));
    expect(screen.getByTestId("open").textContent).toBe("true");
    await act(() => userEvent.click(screen.getByTestId("close-btn")));
    expect(screen.getByTestId("open").textContent).toBe("false");
  });

  it("throws when used outside provider", () => {
    expect(() => render(<TestConsumer />)).toThrow(
      "usePlayerDrawer must be used within PlayerDrawerProvider",
    );
  });
});
