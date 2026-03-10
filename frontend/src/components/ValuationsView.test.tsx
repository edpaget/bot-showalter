import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PlayerDrawerProvider } from "../context/PlayerDrawerContext";
import { VALUATIONS_QUERY } from "../graphql/queries";
import { ValuationsView } from "./ValuationsView";

function valMock(): MockedResponse {
  return {
    request: {
      query: VALUATIONS_QUERY,
      variables: { season: 2026, top: 100 },
    },
    result: {
      data: {
        valuations: [
          {
            playerName: "Mike Trout",
            system: "zar",
            version: "1.0",
            projectionSystem: "steamer",
            projectionVersion: "2026",
            playerType: "batter",
            position: "OF",
            value: 35.0,
            rank: 1,
            categoryScores: { HR: 2.5 },
          },
          {
            playerName: "Gerrit Cole",
            system: "zar",
            version: "1.0",
            projectionSystem: "steamer",
            projectionVersion: "2026",
            playerType: "pitcher",
            position: "SP",
            value: 25.0,
            rank: 2,
            categoryScores: { W: 1.5 },
          },
        ],
      },
    },
  };
}

function renderView(mocks: MockedResponse[] = [valMock()]) {
  return render(
    <MockedProvider mocks={mocks} addTypename={false}>
      <PlayerDrawerProvider>
        <ValuationsView season={2026} />
      </PlayerDrawerProvider>
    </MockedProvider>,
  );
}

describe("ValuationsView", () => {
  afterEach(cleanup);

  it("renders title and filter bar", () => {
    renderView();
    expect(screen.getByText("Valuation Rankings")).toBeInTheDocument();
  });

  it("loads and displays valuations with version", async () => {
    renderView();
    expect(await screen.findByText("Mike Trout")).toBeInTheDocument();
    // Check values and version are displayed
    expect(screen.getByText("$35.0")).toBeInTheDocument();
    expect(screen.getByText("$25.0")).toBeInTheDocument();
    // Version column header and values
    expect(screen.getByText("Version")).toBeInTheDocument();
  });
});
