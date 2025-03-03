from pydantic import BaseModel


class DescriptionForm(BaseModel):
    brand: str = ''
    gender: str = ''
    from_: str = ''
    item_name: str = ''
