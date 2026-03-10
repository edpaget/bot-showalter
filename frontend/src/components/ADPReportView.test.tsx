import { render, screen } from "@testing-library/react";
import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { describe, it, expect } from "vitest";
import { ADPReportView } from "./ADPReportView";
import { PlayerDrawerProvider } from "../context/PlayerDrawerContext";
import { ADP_REPORT_QUERY } from "../graphql/queries";

function adpMock(): MockedResponse {
  return {
    request: {
      query: ADP_REPORT_QUERY,
      variables: { season: 2026 },
    },
    result: {
      data: {
        adpReport: {
          season: 2026,
          system: "zar",
          version: "1.0",
          provider: "fantasypros",
          buyTargets: [
            {
              playerId: 1,
              playerName: "Mike Trout",
              playerType: "batter",
              position: "OF",
              zarRank: 1,
              zarValue: 35.0,
              adpRank: 5,
              adpPick: 5.0,
              rankDelta: 4,
              provider: "fantasypros",
            },
          ],
          avoidList: [
            {
              playerId: 2,
              playerName: "Shohei Ohtani",
              playerType: "batter",
              position: "OF",
              zarRank: 5,
              zarValue: 25.0,
              adpRank: 2,
              adpPick: 2.0,
              rankDelta: -3,
              provider: "fantasypros",
            },
          ],
          unrankedValuable: [],
          nMatched: 3,
        },
      },
    },
  };
}

function renderView() {
  return render(
    <MockedProvider mocks={[adpMock()]} addTypename={false}>
      <PlayerDrawerProvider>
        <ADPReportView season={2026} />
      </PlayerDrawerProvider>
    </MockedProvider>,
  );
}

describe("ADPReportView", () => {
  it("renders report with buy targets, avoid list, and metadata", async () => {
    renderView();

    // Wait for data to load
    expect(await screen.findByText(/Buy Targets/)).toBeInTheDocument();
    expect(screen.getByText(/Avoid List/)).toBeInTheDocument();
    expect(screen.getByText(/Unranked Valuable/)).toBeInTheDocument();

    // Buy target with positive delta
    expect(screen.getByText("Mike Trout")).toBeInTheDocument();

    // Avoid list with negative delta
    expect(screen.getByText("Shohei Ohtani")).toBeInTheDocument();

    // Report metadata
    expect(screen.getByText(/3 matched/)).toBeInTheDocument();
  });
});
