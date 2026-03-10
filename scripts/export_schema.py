"""Export the Strawberry GraphQL schema as SDL to stdout."""

import strawberry

from fantasy_baseball_manager.web.schema import Mutation, Query, Subscription

schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
print(schema.as_str())
