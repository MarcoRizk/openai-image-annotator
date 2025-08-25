from pydantic import BaseModel, Field


class Output(BaseModel):
    description: str = Field(...,
                                    description="A detailed description of what is happening in the image")
    is_outdoor: bool = Field(..., description="Whether the the scene is outdoor or indoor")
    people_count: int = Field(..., description="The number of people who are involved in this scene")
