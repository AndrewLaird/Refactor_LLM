from pydantic_llm_model import PydanticLLMModel
from pydantic import BaseModel, Field
from typing import Literal

class ReportOptions(BaseModel):
    start_month: Literal['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    start_year: int
    end_month: Literal['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    end_year: int
    lease_type: Literal['All', 'Operating', 'Finance']
    region: Literal['All', 'APAC', 'EUROPE', 'LATAM', 'MEA', 'NA']
    separate_by_asset_category: bool
    show_in_base_currency: bool = Field(default=False)
    use_ifrs: bool
    show_split_business_unit_allocations: bool
    show_split_cost_centers: bool
    filter_leases_by_reporting_dates: bool


model = PydanticLLMModel(pydantic_object=ReportOptions)

result = model.invoke("ifrs reporting on sept 12 to october 24")

print(result)
