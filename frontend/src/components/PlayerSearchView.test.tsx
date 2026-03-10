import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { act, cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { PlayerDrawerProvider } from "../context/PlayerDrawerContext";
import { PLAYER_SEARCH_QUERY } from "../graphql/queries";
import { PlayerSearchView } from "./PlayerSearchView";

function searchMock(): MockedResponse {
  return {
    request: {
      query: PLAYER_SEARCH_QUERY,
      variables: { name: "Trout", season: 2026 },
    },
    result: {
      data: {
        playerSearch: [
          {
            playerId: 1,
            name: "Mike Trout",
            team: "LAA",
            age: 34,
            primaryPosition: "CF",
            bats: "R",
            throws: "R",
            experience: 14,
          },
        ],
      },
    },
  };
}

function renderView(mocks: MockedResponse[] = [searchMock()]) {
  return render(
    <MockedProvider mocks={mocks} addTypename={false}>
      <PlayerDrawerProvider>
        <PlayerSearchView season={2026} />
      </PlayerDrawerProvider>
    </MockedProvider>,
  );
}

describe("PlayerSearchView", () => {
  afterEach(cleanup);

  it("renders search form", () => {
    renderView();
    expect(screen.getByText("Player Search")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Search player...")).toBeInTheDocument();
  });

  it("fetches and displays players on submit", async () => {
    renderView();
    const input = screen.getByPlaceholderText("Search player...");
    await act(() => userEvent.type(input, "Trout"));
    await act(() => userEvent.click(screen.getByText("Search")));

    expect(await screen.findByText("Mike Trout")).toBeInTheDocument();
    expect(screen.getByText("LAA")).toBeInTheDocument();
    expect(screen.getByText("CF")).toBeInTheDocument();
    expect(screen.getByText("R/R")).toBeInTheDocument();
  });
});
