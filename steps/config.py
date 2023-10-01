from zenml.steps import BaseParameters

class ModuleNameConfig(BaseParameters):
    """Model Configs"""
    model_name: str = "LinearRegression"