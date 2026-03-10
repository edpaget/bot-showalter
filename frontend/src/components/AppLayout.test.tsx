import { render, screen, cleanup } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { MockedProvider } from "@apollo/client/testing";
import { describe, it, expect, afterEach } from "vitest";
import { AppLayout } from "./AppLayout";
import { PlayerDrawerProvider } from "../context/PlayerDrawerContext";

function renderLayout(initialEntry = "/") {
  return render(
    <MockedProvider mocks={[]} addTypename={false}>
      <PlayerDrawerProvider>
        <MemoryRouter initialEntries={[initialEntry]}>
          <AppLayout />
        </MemoryRouter>
      </PlayerDrawerProvider>
    </MockedProvider>,
  );
}

describe("AppLayout", () => {
  afterEach(cleanup);

  it("renders navigation links", () => {
    renderLayout();
    expect(screen.getByText("Draft")).toBeInTheDocument();
    expect(screen.getByText("Projections")).toBeInTheDocument();
    expect(screen.getByText("Valuations")).toBeInTheDocument();
    expect(screen.getByText("ADP Report")).toBeInTheDocument();
    expect(screen.getByText("Player Search")).toBeInTheDocument();
  });

  it("renders FBM brand", () => {
    renderLayout();
    expect(screen.getByText("FBM")).toBeInTheDocument();
  });

  it("highlights active link", () => {
    renderLayout("/projections");
    const link = screen.getByText("Projections");
    expect(link.className).toContain("bg-blue-600");
  });
});
