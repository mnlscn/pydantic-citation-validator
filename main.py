## Code is already working, good idea to attach an actual retriever from an external KB



from loguru import logger
from pydantic import Field, BaseModel, FieldValidationInfo, model_validator
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class Fact(BaseModel):
    statement: str = Field(
        ..., description="Body of the sentence, as part of a response"
    )

    substring_phrase: list[str] = Field(
        ...,
        description="String quote long enough to evaluate the truthfulness of the fact",
    )

    @model_validator(mode="after")
    def validate_sources(self, info: FieldValidationInfo) -> "Fact":
        """
        For each substring_phrase, find the span of the substring_phrase in the context.
        If the span is not found, remove the substring_phrase from the list.
        """
        if info.context is None:
            logger.info("No context found, skipping validation")
            return self

        # Get the context from the info
        text_chunks = info.context.get("text_chunk", None)

        # Get the spans of the substring_phrase in the context
        spans = list(self.get_spans(text_chunks))
        logger.info(
            f"Found {len(spans)} span(s) for from {len(self.substring_phrase)} citation(s)."
        )
        # Replace the substring_phrase with the actual substring
        self.substring_phrase = [text_chunks[span[0] : span[1]] for span in spans]
        return self

    def _get_span(self, quote, context, errs=5):
        import regex

        minor = quote
        major = context

        errs_ = 0
        s = regex.search(f"({minor}){{e<={errs_}}}", major)
        while s is None and errs_ <= errs:
            errs_ += 1
            s = regex.search(f"({minor}){{e<={errs_}}}", major)

        if s is not None:
            yield from s.spans()

    def get_spans(self, context):
        for quote in self.substring_phrase:
            yield from self._get_span(quote, context)


class QuestionAnswer(BaseModel):
    """
    Class representing a question and its answer as a list of facts each one should have a soruce.
    each sentence contains a body and a list of sources."""

    question: str = Field(..., description="Question that was asked")
    answer: list[Fact] = Field(
        ...,
        description="Body of the answer, each fact should be its seperate object with a body and a list of sources",
    )

    @model_validator(mode="after")
    def validate_sources(self) -> "QuestionAnswer":
        """
        Checks that each fact has some sources, and removes those that do not.
        """
        logger.info(f"Validating {len(self.answer)} facts")
        self.answer = [fact for fact in self.answer if len(fact.substring_phrase) > 0]
        logger.info(f"Found {len(self.answer)} facts with sources")
        return self


question_input = "where did he go to school?"
context_input = """
My name is Jason Liu, and I grew up in Toronto Canada but I was born in China.I went to an arts highschool but in university I studied Computational Mathematics and physics.  As part of coop I worked at many companies including Stitchfix, Facebook. I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.
"""


def ask_ai(question: str, context: str) -> QuestionAnswer:

    prompt = PromptTemplate.from_template(
        """
                            
    You are a world class algorithm to answer questions with correct and exact citations.

    # CONTEXT:
    {context}

    # QUESTION:
    {question}
                                
    """
        )

    formatted_prompt = prompt.format(context=context_input, question=question_input)
    print(formatted_prompt)
    llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
        QuestionAnswer
    )

    result = llm.invoke(formatted_prompt)
    return result

answer = ask_ai(question=question_input, context=context_input)
print(answer.model_dump_json(indent=2))
