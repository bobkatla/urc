from pydantic import BaseModel, Field

class PhiConfig(BaseModel):
    d_out: int = Field(..., description="Backbone feature dim before PCA")
    pca_k: int = Field(..., description="PCA output dimension")
