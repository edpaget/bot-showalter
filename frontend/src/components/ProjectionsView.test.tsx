import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { act, cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { PlayerDrawerProvider } from "../context/PlayerDrawerContext";
import { PROJECTIONS_QUERY } from "../graphql/queries";
import { ProjectionsView } from "./ProjectionsView";

function projMock(): MockedResponse {
  return {
    request: {
      query: PROJECTIONS_QUERY,
      variables: { season: 2026, playerName: "Trout" },
    },
    result: {
      data: {
        projections: [
          {
            playerName: "Mike Trout",
            system: "steamer",
            version: "2026",
            sourceType: "first_party",
            playerType: "batter",
            stats: { pa: 600, hr: 35, rbi: 90 },
          },
        ],
      },
    },
  };
}

function renderView(mocks: MockedResponse[] = [projMock()]) {
  return render(
    <MockedProvider mocks={mocks} addTypename={false}>
      <PlayerDrawerProvider>
        <ProjectionsView season={2026} />
      </PlayerDrawerProvider>
    </MockedProvider>,
  );
}

describe("ProjectionsView", () => {
  afterEach(cleanup);

  it("renders search form", () => {
    renderView();
    expect(screen.getByText("Projections Lookup")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Search player...")).toBeInTheDocument();
  });

  it("fetches and displays projections on submit", async () => {
    renderView();
    const input = screen.getByPlaceholderText("Search player...");
    await act(() => userEvent.type(input, "Trout"));
    await act(() => userEvent.click(screen.getByText("Search")));

    // Wait for results
    expect(await screen.findByText("Mike Trout")).toBeInTheDocument();
    expect(screen.getByText("steamer")).toBeInTheDocument();
    expect(screen.getByText("2026")).toBeInTheDocument();
    expect(screen.getByText("batter")).toBeInTheDocument();
  });
});
