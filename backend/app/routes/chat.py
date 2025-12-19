import traceback
from fastapi import APIRouter, Depends, HTTPException
from ..services.llm_text_formatter import format_llm_response

from ..schemas import ChatRequest, ChatResponse, ParseQueryRequest, ParseQueryResponse
from ..services.rag_service import RAGService, get_rag_service
from ..services.query_parser import QueryParser, get_query_parser

router = APIRouter()


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    try:
        raw_response = rag_service.answer(request)
        raw_response.answer = format_llm_response(raw_response.answer)
        return raw_response
    except Exception as e:
        # Log the full error for debugging
        error_trace = traceback.format_exc()
        print(f"ERROR in chat endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        
        # Return a proper error response instead of letting it bubble up as 500
        return ChatResponse(
            answer=f"I encountered an error while processing your request: {str(e)}. Please try again or rephrase your question.",
            citations=[],
            raw_context=None,
            model=None,
            usage=None,
            retrieval_debug={"error": str(e), "error_type": type(e).__name__},
        )


@router.post("/parse-query", response_model=ParseQueryResponse)
def parse_query(
    request: ParseQueryRequest,
    query_parser: QueryParser = Depends(get_query_parser),
) -> ParseQueryResponse:
    """Parse a user query to extract ticker symbols and time periods."""
    tickers, period, needs_clarification, clarification_message = query_parser.parse(
        request.question
    )
    return ParseQueryResponse(
        tickers=tickers,
        period=period,
        needs_clarification=needs_clarification,
        clarification_message=clarification_message,
    )



