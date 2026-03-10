import type { CodegenConfig } from "@graphql-codegen/cli";

const config: CodegenConfig = {
  schema: "schema.graphql",
  documents: ["src/graphql/documents/**/*.ts"],
  generates: {
    "src/generated/graphql.ts": {
      plugins: ["typescript", "typescript-operations", "typed-document-node"],
      config: {
        enumsAsTypes: true,
        scalars: {
          JSON: "Record<string, unknown>",
        },
        avoidOptionals: {
          field: true,
          object: true,
          inputValue: false,
        },
      },
    },
  },
};

export default config;
