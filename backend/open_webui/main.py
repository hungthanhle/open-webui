import asyncio
import inspect
import json
import logging
import mimetypes
import os
import shutil
import sys
import time
import random

from contextlib import asynccontextmanager
from urllib.parse import urlencode, parse_qs, urlparse
from pydantic import BaseModel
from sqlalchemy import text

from typing import Optional
from aiocache import cached
import aiohttp
import requests


from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
    applications,
    BackgroundTasks,
)

from fastapi.openapi.docs import get_swagger_ui_html

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response, StreamingResponse


from open_webui.socket.main import (
    app as socket_app,
    periodic_usage_pool_cleanup,
)
from open_webui.routers import (
    audio,
    images,
    ollama,
    openai,
    retrieval,
    pipelines,
    tasks,
    auths,
    channels,
    chats,
    folders,
    configs,
    groups,
    files,
    functions,
    memories,
    models,
    knowledge,
    prompts,
    evaluations,
    tools,
    users,
    utils,
)

from open_webui.routers.retrieval import (
    get_embedding_function,
    get_ef,
    get_rf,
)

from open_webui.internal.db import Session

from open_webui.models.functions import Functions
from open_webui.models.models import Models
from open_webui.models.users import UserModel, Users

from open_webui.config import (
    # Ollama
    ENABLE_OLLAMA_API,
    OLLAMA_BASE_URLS,
    OLLAMA_API_CONFIGS,
    # OpenAI
    ENABLE_OPENAI_API,
    OPENAI_API_BASE_URLS,
    OPENAI_API_KEYS,
    OPENAI_API_CONFIGS,
    # Image
    AUTOMATIC1111_API_AUTH,
    AUTOMATIC1111_BASE_URL,
    AUTOMATIC1111_CFG_SCALE,
    AUTOMATIC1111_SAMPLER,
    AUTOMATIC1111_SCHEDULER,
    COMFYUI_BASE_URL,
    COMFYUI_API_KEY,
    COMFYUI_WORKFLOW,
    COMFYUI_WORKFLOW_NODES,
    ENABLE_IMAGE_GENERATION,
    ENABLE_IMAGE_PROMPT_GENERATION,
    IMAGE_GENERATION_ENGINE,
    IMAGE_GENERATION_MODEL,
    IMAGE_SIZE,
    IMAGE_STEPS,
    IMAGES_OPENAI_API_BASE_URL,
    IMAGES_OPENAI_API_KEY,
    # Audio
    AUDIO_STT_ENGINE,
    AUDIO_STT_MODEL,
    AUDIO_STT_OPENAI_API_BASE_URL,
    AUDIO_STT_OPENAI_API_KEY,
    AUDIO_TTS_API_KEY,
    AUDIO_TTS_ENGINE,
    AUDIO_TTS_MODEL,
    AUDIO_TTS_OPENAI_API_BASE_URL,
    AUDIO_TTS_OPENAI_API_KEY,
    AUDIO_TTS_SPLIT_ON,
    AUDIO_TTS_VOICE,
    AUDIO_TTS_AZURE_SPEECH_REGION,
    AUDIO_TTS_AZURE_SPEECH_OUTPUT_FORMAT,
    WHISPER_MODEL,
    WHISPER_MODEL_AUTO_UPDATE,
    WHISPER_MODEL_DIR,
    # Retrieval
    RAG_TEMPLATE,
    DEFAULT_RAG_TEMPLATE,
    RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
    RAG_RERANKING_MODEL,
    RAG_RERANKING_MODEL_AUTO_UPDATE,
    RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
    RAG_EMBEDDING_ENGINE,
    RAG_EMBEDDING_BATCH_SIZE,
    RAG_RELEVANCE_THRESHOLD,
    RAG_FILE_MAX_COUNT,
    RAG_FILE_MAX_SIZE,
    RAG_OPENAI_API_BASE_URL,
    RAG_OPENAI_API_KEY,
    RAG_OLLAMA_BASE_URL,
    RAG_OLLAMA_API_KEY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTENT_EXTRACTION_ENGINE,
    TIKA_SERVER_URL,
    RAG_TOP_K,
    RAG_TEXT_SPLITTER,
    TIKTOKEN_ENCODING_NAME,
    PDF_EXTRACT_IMAGES,
    YOUTUBE_LOADER_LANGUAGE,
    YOUTUBE_LOADER_PROXY_URL,
    # Retrieval (Web Search)
    RAG_WEB_SEARCH_ENGINE,
    RAG_WEB_SEARCH_RESULT_COUNT,
    RAG_WEB_SEARCH_CONCURRENT_REQUESTS,
    RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
    JINA_API_KEY,
    SEARCHAPI_API_KEY,
    SEARCHAPI_ENGINE,
    SEARXNG_QUERY_URL,
    SERPER_API_KEY,
    SERPLY_API_KEY,
    SERPSTACK_API_KEY,
    SERPSTACK_HTTPS,
    TAVILY_API_KEY,
    BING_SEARCH_V7_ENDPOINT,
    BING_SEARCH_V7_SUBSCRIPTION_KEY,
    BRAVE_SEARCH_API_KEY,
    KAGI_SEARCH_API_KEY,
    MOJEEK_SEARCH_API_KEY,
    GOOGLE_PSE_API_KEY,
    GOOGLE_PSE_ENGINE_ID,
    GOOGLE_DRIVE_CLIENT_ID,
    GOOGLE_DRIVE_API_KEY,
    ENABLE_RAG_HYBRID_SEARCH,
    ENABLE_RAG_LOCAL_WEB_FETCH,
    ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
    ENABLE_RAG_WEB_SEARCH,
    ENABLE_GOOGLE_DRIVE_INTEGRATION,
    UPLOAD_DIR,
    # WebUI
    WEBUI_AUTH,
    WEBUI_NAME,
    WEBUI_BANNERS,
    WEBHOOK_URL,
    ADMIN_EMAIL,
    SHOW_ADMIN_DETAILS,
    JWT_EXPIRES_IN,
    ENABLE_SIGNUP,
    ENABLE_LOGIN_FORM,
    ENABLE_API_KEY,
    ENABLE_API_KEY_ENDPOINT_RESTRICTIONS,
    API_KEY_ALLOWED_ENDPOINTS,
    ENABLE_CHANNELS,
    ENABLE_COMMUNITY_SHARING,
    ENABLE_MESSAGE_RATING,
    ENABLE_EVALUATION_ARENA_MODELS,
    USER_PERMISSIONS,
    DEFAULT_USER_ROLE,
    DEFAULT_PROMPT_SUGGESTIONS,
    DEFAULT_MODELS,
    DEFAULT_ARENA_MODEL,
    MODEL_ORDER_LIST,
    EVALUATION_ARENA_MODELS,
    # WebUI (OAuth)
    ENABLE_OAUTH_ROLE_MANAGEMENT,
    OAUTH_ROLES_CLAIM,
    OAUTH_EMAIL_CLAIM,
    OAUTH_PICTURE_CLAIM,
    OAUTH_USERNAME_CLAIM,
    OAUTH_ALLOWED_ROLES,
    OAUTH_ADMIN_ROLES,
    # WebUI (LDAP)
    ENABLE_LDAP,
    LDAP_SERVER_LABEL,
    LDAP_SERVER_HOST,
    LDAP_SERVER_PORT,
    LDAP_ATTRIBUTE_FOR_MAIL,
    LDAP_ATTRIBUTE_FOR_USERNAME,
    LDAP_SEARCH_FILTERS,
    LDAP_SEARCH_BASE,
    LDAP_APP_DN,
    LDAP_APP_PASSWORD,
    LDAP_USE_TLS,
    LDAP_CA_CERT_FILE,
    LDAP_CIPHERS,
    # Misc
    ENV,
    CACHE_DIR,
    STATIC_DIR,
    FRONTEND_BUILD_DIR,
    CORS_ALLOW_ORIGIN,
    DEFAULT_LOCALE,
    OAUTH_PROVIDERS,
    WEBUI_URL,
    # Admin
    ENABLE_ADMIN_CHAT_ACCESS,
    ENABLE_ADMIN_EXPORT,
    # Tasks
    TASK_MODEL,
    TASK_MODEL_EXTERNAL,
    ENABLE_TAGS_GENERATION,
    ENABLE_SEARCH_QUERY_GENERATION,
    ENABLE_RETRIEVAL_QUERY_GENERATION,
    ENABLE_AUTOCOMPLETE_GENERATION,
    TITLE_GENERATION_PROMPT_TEMPLATE,
    TAGS_GENERATION_PROMPT_TEMPLATE,
    IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE,
    TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE,
    QUERY_GENERATION_PROMPT_TEMPLATE,
    AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE,
    AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH,
    AppConfig,
    reset_config,
)
from open_webui.env import (
    CHANGELOG,
    GLOBAL_LOG_LEVEL,
    SAFE_MODE,
    SRC_LOG_LEVELS,
    VERSION,
    WEBUI_BUILD_HASH,
    WEBUI_SECRET_KEY,
    WEBUI_SESSION_COOKIE_SAME_SITE,
    WEBUI_SESSION_COOKIE_SECURE,
    WEBUI_AUTH_TRUSTED_EMAIL_HEADER,
    WEBUI_AUTH_TRUSTED_NAME_HEADER,
    ENABLE_WEBSOCKET_SUPPORT,
    BYPASS_MODEL_ACCESS_CONTROL,
    RESET_CONFIG_ON_START,
    OFFLINE_MODE,
)


from open_webui.utils.models import (
    get_all_models,
    get_all_base_models,
    check_model_access,
)
from open_webui.utils.chat import (
    generate_chat_completion as chat_completion_handler,
    chat_completed as chat_completed_handler,
    chat_action as chat_action_handler,
)
from open_webui.utils.middleware import process_chat_payload, process_chat_response
from open_webui.utils.access_control import has_access

from open_webui.utils.auth import (
    decode_token,
    get_admin_user,
    get_verified_user,
)
from open_webui.utils.oauth import oauth_manager
from open_webui.utils.security_headers import SecurityHeadersMiddleware

from open_webui.tasks import stop_task, list_tasks  # Import from tasks.py

if SAFE_MODE:
    print("SAFE MODE ENABLED")
    Functions.deactivate_all_functions()

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except (HTTPException, StarletteHTTPException) as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex


print(
    rf"""
  ___                    __        __   _     _   _ ___
 / _ \ _ __   ___ _ __   \ \      / /__| |__ | | | |_ _|
| | | | '_ \ / _ \ '_ \   \ \ /\ / / _ \ '_ \| | | || |
| |_| | |_) |  __/ | | |   \ V  V /  __/ |_) | |_| || |
 \___/| .__/ \___|_| |_|    \_/\_/ \___|_.__/ \___/|___|
      |_|


v{VERSION} - building the best open-source AI user interface.
{f"Commit: {WEBUI_BUILD_HASH}" if WEBUI_BUILD_HASH != "dev-build" else ""}
https://github.com/open-webui/open-webui
"""
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if RESET_CONFIG_ON_START:
        reset_config()

    asyncio.create_task(periodic_usage_pool_cleanup())
    yield


app = FastAPI(
    docs_url="/docs" if ENV == "dev" else None,
    openapi_url="/openapi.json" if ENV == "dev" else None,
    redoc_url=None,
    lifespan=lifespan,
)

app.state.config = AppConfig()


########################################
#
# OLLAMA
#
########################################


app.state.config.ENABLE_OLLAMA_API = ENABLE_OLLAMA_API
app.state.config.OLLAMA_BASE_URLS = OLLAMA_BASE_URLS
app.state.config.OLLAMA_API_CONFIGS = OLLAMA_API_CONFIGS

app.state.OLLAMA_MODELS = {}

########################################
#
# OPENAI
#
########################################

app.state.config.ENABLE_OPENAI_API = ENABLE_OPENAI_API
app.state.config.OPENAI_API_BASE_URLS = OPENAI_API_BASE_URLS
app.state.config.OPENAI_API_KEYS = OPENAI_API_KEYS
app.state.config.OPENAI_API_CONFIGS = OPENAI_API_CONFIGS

app.state.OPENAI_MODELS = {}

########################################
#
# WEBUI
#
########################################

app.state.config.WEBUI_URL = WEBUI_URL
app.state.config.ENABLE_SIGNUP = ENABLE_SIGNUP
app.state.config.ENABLE_LOGIN_FORM = ENABLE_LOGIN_FORM

app.state.config.ENABLE_API_KEY = ENABLE_API_KEY
app.state.config.ENABLE_API_KEY_ENDPOINT_RESTRICTIONS = (
    ENABLE_API_KEY_ENDPOINT_RESTRICTIONS
)
app.state.config.API_KEY_ALLOWED_ENDPOINTS = API_KEY_ALLOWED_ENDPOINTS

app.state.config.JWT_EXPIRES_IN = JWT_EXPIRES_IN

app.state.config.SHOW_ADMIN_DETAILS = SHOW_ADMIN_DETAILS
app.state.config.ADMIN_EMAIL = ADMIN_EMAIL


app.state.config.DEFAULT_MODELS = DEFAULT_MODELS
app.state.config.DEFAULT_PROMPT_SUGGESTIONS = DEFAULT_PROMPT_SUGGESTIONS
app.state.config.DEFAULT_USER_ROLE = DEFAULT_USER_ROLE

app.state.config.USER_PERMISSIONS = USER_PERMISSIONS
app.state.config.WEBHOOK_URL = WEBHOOK_URL
app.state.config.BANNERS = WEBUI_BANNERS
app.state.config.MODEL_ORDER_LIST = MODEL_ORDER_LIST


app.state.config.ENABLE_CHANNELS = ENABLE_CHANNELS
app.state.config.ENABLE_COMMUNITY_SHARING = ENABLE_COMMUNITY_SHARING
app.state.config.ENABLE_MESSAGE_RATING = ENABLE_MESSAGE_RATING

app.state.config.ENABLE_EVALUATION_ARENA_MODELS = ENABLE_EVALUATION_ARENA_MODELS
app.state.config.EVALUATION_ARENA_MODELS = EVALUATION_ARENA_MODELS

app.state.config.OAUTH_USERNAME_CLAIM = OAUTH_USERNAME_CLAIM
app.state.config.OAUTH_PICTURE_CLAIM = OAUTH_PICTURE_CLAIM
app.state.config.OAUTH_EMAIL_CLAIM = OAUTH_EMAIL_CLAIM

app.state.config.ENABLE_OAUTH_ROLE_MANAGEMENT = ENABLE_OAUTH_ROLE_MANAGEMENT
app.state.config.OAUTH_ROLES_CLAIM = OAUTH_ROLES_CLAIM
app.state.config.OAUTH_ALLOWED_ROLES = OAUTH_ALLOWED_ROLES
app.state.config.OAUTH_ADMIN_ROLES = OAUTH_ADMIN_ROLES

app.state.config.ENABLE_LDAP = ENABLE_LDAP
app.state.config.LDAP_SERVER_LABEL = LDAP_SERVER_LABEL
app.state.config.LDAP_SERVER_HOST = LDAP_SERVER_HOST
app.state.config.LDAP_SERVER_PORT = LDAP_SERVER_PORT
app.state.config.LDAP_ATTRIBUTE_FOR_MAIL = LDAP_ATTRIBUTE_FOR_MAIL
app.state.config.LDAP_ATTRIBUTE_FOR_USERNAME = LDAP_ATTRIBUTE_FOR_USERNAME
app.state.config.LDAP_APP_DN = LDAP_APP_DN
app.state.config.LDAP_APP_PASSWORD = LDAP_APP_PASSWORD
app.state.config.LDAP_SEARCH_BASE = LDAP_SEARCH_BASE
app.state.config.LDAP_SEARCH_FILTERS = LDAP_SEARCH_FILTERS
app.state.config.LDAP_USE_TLS = LDAP_USE_TLS
app.state.config.LDAP_CA_CERT_FILE = LDAP_CA_CERT_FILE
app.state.config.LDAP_CIPHERS = LDAP_CIPHERS


app.state.AUTH_TRUSTED_EMAIL_HEADER = WEBUI_AUTH_TRUSTED_EMAIL_HEADER
app.state.AUTH_TRUSTED_NAME_HEADER = WEBUI_AUTH_TRUSTED_NAME_HEADER

app.state.TOOLS = {}
app.state.FUNCTIONS = {}


########################################
#
# RETRIEVAL
#
########################################


app.state.config.TOP_K = RAG_TOP_K
app.state.config.RELEVANCE_THRESHOLD = RAG_RELEVANCE_THRESHOLD
app.state.config.FILE_MAX_SIZE = RAG_FILE_MAX_SIZE
app.state.config.FILE_MAX_COUNT = RAG_FILE_MAX_COUNT

app.state.config.ENABLE_RAG_HYBRID_SEARCH = ENABLE_RAG_HYBRID_SEARCH
app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION = (
    ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION
)

app.state.config.CONTENT_EXTRACTION_ENGINE = CONTENT_EXTRACTION_ENGINE
app.state.config.TIKA_SERVER_URL = TIKA_SERVER_URL

app.state.config.TEXT_SPLITTER = RAG_TEXT_SPLITTER
app.state.config.TIKTOKEN_ENCODING_NAME = TIKTOKEN_ENCODING_NAME

app.state.config.CHUNK_SIZE = CHUNK_SIZE
app.state.config.CHUNK_OVERLAP = CHUNK_OVERLAP

app.state.config.RAG_EMBEDDING_ENGINE = RAG_EMBEDDING_ENGINE
app.state.config.RAG_EMBEDDING_MODEL = RAG_EMBEDDING_MODEL
app.state.config.RAG_EMBEDDING_BATCH_SIZE = RAG_EMBEDDING_BATCH_SIZE
app.state.config.RAG_RERANKING_MODEL = RAG_RERANKING_MODEL
app.state.config.RAG_TEMPLATE = RAG_TEMPLATE

app.state.config.RAG_OPENAI_API_BASE_URL = RAG_OPENAI_API_BASE_URL
app.state.config.RAG_OPENAI_API_KEY = RAG_OPENAI_API_KEY

app.state.config.RAG_OLLAMA_BASE_URL = RAG_OLLAMA_BASE_URL
app.state.config.RAG_OLLAMA_API_KEY = RAG_OLLAMA_API_KEY

app.state.config.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

app.state.config.YOUTUBE_LOADER_LANGUAGE = YOUTUBE_LOADER_LANGUAGE
app.state.config.YOUTUBE_LOADER_PROXY_URL = YOUTUBE_LOADER_PROXY_URL


app.state.config.ENABLE_RAG_WEB_SEARCH = ENABLE_RAG_WEB_SEARCH
app.state.config.RAG_WEB_SEARCH_ENGINE = RAG_WEB_SEARCH_ENGINE
app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST = RAG_WEB_SEARCH_DOMAIN_FILTER_LIST

app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION = ENABLE_GOOGLE_DRIVE_INTEGRATION
app.state.config.SEARXNG_QUERY_URL = SEARXNG_QUERY_URL
app.state.config.GOOGLE_PSE_API_KEY = GOOGLE_PSE_API_KEY
app.state.config.GOOGLE_PSE_ENGINE_ID = GOOGLE_PSE_ENGINE_ID
app.state.config.BRAVE_SEARCH_API_KEY = BRAVE_SEARCH_API_KEY
app.state.config.KAGI_SEARCH_API_KEY = KAGI_SEARCH_API_KEY
app.state.config.MOJEEK_SEARCH_API_KEY = MOJEEK_SEARCH_API_KEY
app.state.config.SERPSTACK_API_KEY = SERPSTACK_API_KEY
app.state.config.SERPSTACK_HTTPS = SERPSTACK_HTTPS
app.state.config.SERPER_API_KEY = SERPER_API_KEY
app.state.config.SERPLY_API_KEY = SERPLY_API_KEY
app.state.config.TAVILY_API_KEY = TAVILY_API_KEY
app.state.config.SEARCHAPI_API_KEY = SEARCHAPI_API_KEY
app.state.config.SEARCHAPI_ENGINE = SEARCHAPI_ENGINE
app.state.config.JINA_API_KEY = JINA_API_KEY
app.state.config.BING_SEARCH_V7_ENDPOINT = BING_SEARCH_V7_ENDPOINT
app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY = BING_SEARCH_V7_SUBSCRIPTION_KEY

app.state.config.RAG_WEB_SEARCH_RESULT_COUNT = RAG_WEB_SEARCH_RESULT_COUNT
app.state.config.RAG_WEB_SEARCH_CONCURRENT_REQUESTS = RAG_WEB_SEARCH_CONCURRENT_REQUESTS

app.state.EMBEDDING_FUNCTION = None
app.state.ef = None
app.state.rf = None

app.state.YOUTUBE_LOADER_TRANSLATION = None


try:
    app.state.ef = get_ef(
        app.state.config.RAG_EMBEDDING_ENGINE,
        app.state.config.RAG_EMBEDDING_MODEL,
        RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    )

    app.state.rf = get_rf(
        app.state.config.RAG_RERANKING_MODEL,
        RAG_RERANKING_MODEL_AUTO_UPDATE,
    )
except Exception as e:
    log.error(f"Error updating models: {e}")
    pass


app.state.EMBEDDING_FUNCTION = get_embedding_function(
    app.state.config.RAG_EMBEDDING_ENGINE,
    app.state.config.RAG_EMBEDDING_MODEL,
    app.state.ef,
    (
        app.state.config.RAG_OPENAI_API_BASE_URL
        if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
        else app.state.config.RAG_OLLAMA_BASE_URL
    ),
    (
        app.state.config.RAG_OPENAI_API_KEY
        if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
        else app.state.config.RAG_OLLAMA_API_KEY
    ),
    app.state.config.RAG_EMBEDDING_BATCH_SIZE,
)


########################################
#
# IMAGES
#
########################################

app.state.config.IMAGE_GENERATION_ENGINE = IMAGE_GENERATION_ENGINE
app.state.config.ENABLE_IMAGE_GENERATION = ENABLE_IMAGE_GENERATION
app.state.config.ENABLE_IMAGE_PROMPT_GENERATION = ENABLE_IMAGE_PROMPT_GENERATION

app.state.config.IMAGES_OPENAI_API_BASE_URL = IMAGES_OPENAI_API_BASE_URL
app.state.config.IMAGES_OPENAI_API_KEY = IMAGES_OPENAI_API_KEY

app.state.config.IMAGE_GENERATION_MODEL = IMAGE_GENERATION_MODEL

app.state.config.AUTOMATIC1111_BASE_URL = AUTOMATIC1111_BASE_URL
app.state.config.AUTOMATIC1111_API_AUTH = AUTOMATIC1111_API_AUTH
app.state.config.AUTOMATIC1111_CFG_SCALE = AUTOMATIC1111_CFG_SCALE
app.state.config.AUTOMATIC1111_SAMPLER = AUTOMATIC1111_SAMPLER
app.state.config.AUTOMATIC1111_SCHEDULER = AUTOMATIC1111_SCHEDULER
app.state.config.COMFYUI_BASE_URL = COMFYUI_BASE_URL
app.state.config.COMFYUI_API_KEY = COMFYUI_API_KEY
app.state.config.COMFYUI_WORKFLOW = COMFYUI_WORKFLOW
app.state.config.COMFYUI_WORKFLOW_NODES = COMFYUI_WORKFLOW_NODES

app.state.config.IMAGE_SIZE = IMAGE_SIZE
app.state.config.IMAGE_STEPS = IMAGE_STEPS


########################################
#
# AUDIO
#
########################################

app.state.config.STT_OPENAI_API_BASE_URL = AUDIO_STT_OPENAI_API_BASE_URL
app.state.config.STT_OPENAI_API_KEY = AUDIO_STT_OPENAI_API_KEY
app.state.config.STT_ENGINE = AUDIO_STT_ENGINE
app.state.config.STT_MODEL = AUDIO_STT_MODEL

app.state.config.WHISPER_MODEL = WHISPER_MODEL

app.state.config.TTS_OPENAI_API_BASE_URL = AUDIO_TTS_OPENAI_API_BASE_URL
app.state.config.TTS_OPENAI_API_KEY = AUDIO_TTS_OPENAI_API_KEY
app.state.config.TTS_ENGINE = AUDIO_TTS_ENGINE
app.state.config.TTS_MODEL = AUDIO_TTS_MODEL
app.state.config.TTS_VOICE = AUDIO_TTS_VOICE
app.state.config.TTS_API_KEY = AUDIO_TTS_API_KEY
app.state.config.TTS_SPLIT_ON = AUDIO_TTS_SPLIT_ON


app.state.config.TTS_AZURE_SPEECH_REGION = AUDIO_TTS_AZURE_SPEECH_REGION
app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT = AUDIO_TTS_AZURE_SPEECH_OUTPUT_FORMAT


app.state.faster_whisper_model = None
app.state.speech_synthesiser = None
app.state.speech_speaker_embeddings_dataset = None


########################################
#
# TASKS
#
########################################


app.state.config.TASK_MODEL = TASK_MODEL
app.state.config.TASK_MODEL_EXTERNAL = TASK_MODEL_EXTERNAL


app.state.config.ENABLE_SEARCH_QUERY_GENERATION = ENABLE_SEARCH_QUERY_GENERATION
app.state.config.ENABLE_RETRIEVAL_QUERY_GENERATION = ENABLE_RETRIEVAL_QUERY_GENERATION
app.state.config.ENABLE_AUTOCOMPLETE_GENERATION = ENABLE_AUTOCOMPLETE_GENERATION
app.state.config.ENABLE_TAGS_GENERATION = ENABLE_TAGS_GENERATION


app.state.config.TITLE_GENERATION_PROMPT_TEMPLATE = TITLE_GENERATION_PROMPT_TEMPLATE
app.state.config.TAGS_GENERATION_PROMPT_TEMPLATE = TAGS_GENERATION_PROMPT_TEMPLATE
app.state.config.IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE = (
    IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE
)

app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE = (
    TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE
)
app.state.config.QUERY_GENERATION_PROMPT_TEMPLATE = QUERY_GENERATION_PROMPT_TEMPLATE
app.state.config.AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE = (
    AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE
)
app.state.config.AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH = (
    AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH
)


########################################
#
# WEBUI
#
########################################

app.state.MODELS = {}


class RedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Check if the request is a GET request
        if request.method == "GET":
            path = request.url.path
            query_params = dict(parse_qs(urlparse(str(request.url)).query))

            # Check for the specific watch path and the presence of 'v' parameter
            if path.endswith("/watch") and "v" in query_params:
                video_id = query_params["v"][0]  # Extract the first 'v' parameter
                encoded_video_id = urlencode({"youtube": video_id})
                redirect_url = f"/?{encoded_video_id}"
                return RedirectResponse(url=redirect_url)

        # Proceed with the normal flow of other requests
        response = await call_next(request)
        return response


# Add the middleware to the app
app.add_middleware(RedirectMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


@app.middleware("http")
async def commit_session_after_request(request: Request, call_next):
    response = await call_next(request)
    # log.debug("Commit session after request")
    Session.commit()
    return response


@app.middleware("http")
async def check_url(request: Request, call_next):
    start_time = int(time.time())
    request.state.enable_api_key = app.state.config.ENABLE_API_KEY
    response = await call_next(request)
    process_time = int(time.time()) - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def inspect_websocket(request: Request, call_next):
    if (
        "/ws/socket.io" in request.url.path
        and request.query_params.get("transport") == "websocket"
    ):
        upgrade = (request.headers.get("Upgrade") or "").lower()
        connection = (request.headers.get("Connection") or "").lower().split(",")
        # Check that there's the correct headers for an upgrade, else reject the connection
        # This is to work around this upstream issue: https://github.com/miguelgrinberg/python-engineio/issues/367
        if upgrade != "websocket" or "upgrade" not in connection:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid WebSocket upgrade request"},
            )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGIN,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/ws", socket_app)


app.include_router(ollama.router, prefix="/ollama", tags=["ollama"])
app.include_router(openai.router, prefix="/openai", tags=["openai"])


app.include_router(pipelines.router, prefix="/api/v1/pipelines", tags=["pipelines"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["tasks"])
app.include_router(images.router, prefix="/api/v1/images", tags=["images"])
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(retrieval.router, prefix="/api/v1/retrieval", tags=["retrieval"])

app.include_router(configs.router, prefix="/api/v1/configs", tags=["configs"])

app.include_router(auths.router, prefix="/api/v1/auths", tags=["auths"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])


app.include_router(channels.router, prefix="/api/v1/channels", tags=["channels"])
app.include_router(chats.router, prefix="/api/v1/chats", tags=["chats"])

app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(prompts.router, prefix="/api/v1/prompts", tags=["prompts"])
app.include_router(tools.router, prefix="/api/v1/tools", tags=["tools"])

app.include_router(memories.router, prefix="/api/v1/memories", tags=["memories"])
app.include_router(folders.router, prefix="/api/v1/folders", tags=["folders"])
app.include_router(groups.router, prefix="/api/v1/groups", tags=["groups"])
app.include_router(files.router, prefix="/api/v1/files", tags=["files"])
app.include_router(functions.router, prefix="/api/v1/functions", tags=["functions"])
app.include_router(
    evaluations.router, prefix="/api/v1/evaluations", tags=["evaluations"]
)
app.include_router(utils.router, prefix="/api/v1/utils", tags=["utils"])


##################################
#
# Chat Endpoints
#
##################################


@app.get("/api/models")
async def get_models(request: Request, user=Depends(get_verified_user)):
    def get_filtered_models(models, user):
        filtered_models = []
        for model in models:
            if model.get("arena"):
                if has_access(
                    user.id,
                    type="read",
                    access_control=model.get("info", {})
                    .get("meta", {})
                    .get("access_control", {}),
                ):
                    filtered_models.append(model)
                continue

            model_info = Models.get_model_by_id(model["id"])
            if model_info:
                if user.id == model_info.user_id or has_access(
                    user.id, type="read", access_control=model_info.access_control
                ):
                    filtered_models.append(model)

        return filtered_models

    models = await get_all_models(request)

    # Filter out filter pipelines
    models = [
        model
        for model in models
        if "pipeline" not in model or model["pipeline"].get("type", None) != "filter"
    ]

    model_order_list = request.app.state.config.MODEL_ORDER_LIST
    if model_order_list:
        model_order_dict = {model_id: i for i, model_id in enumerate(model_order_list)}
        # Sort models by order list priority, with fallback for those not in the list
        models.sort(
            key=lambda x: (model_order_dict.get(x["id"], float("inf")), x["name"])
        )

    # Filter out models that the user does not have access to
    if user.role == "user" and not BYPASS_MODEL_ACCESS_CONTROL:
        models = get_filtered_models(models, user)

    log.debug(
        f"/api/models returned filtered models accessible to the user: {json.dumps([model['id'] for model in models])}"
    )
    return {"data": models}


@app.get("/api/models/base")
async def get_base_models(request: Request, user=Depends(get_admin_user)):
    models = await get_all_base_models(request)
    return {"data": models}

'''
request

{
    "stream": true,
    "model": "deepseek-r1:1.5b",
    "messages": [
        {
            "role": "user",
            "content": "chat title"
        },
        {
            "role": "assistant",
            "content": "Sure! Could you clarify or expand on your request? I'm here to help with any questions or topics you'd like to discuss. What's the focus of your chat, and are there specific areas you'd like to explore?"
        },
        {
            "role": "user",
            "content": "bạn lưu tin nhắn ở đâu trong database thế"
        },
        {
            "role": "assistant",
            "content": "Tôi không có thông tin cụ thể về cách lưu tinymce trong một base of données. Tuy nhiên, có thể có nhiều cách để sau diện vào file tinymce trong một base of data khác:\n\n1. **Tùy chọn file XML**: Nếu bạn sử dụng base of data XML (tuy nhiên base of data này không phải là base of data web), có thể đã có một base of data XML được sử dụng để lưu các phần tử HTML và CSS của tinymce, sau đó bạn sẽ tạo một base of data mới để lưu các file files JavaScript và styles của tinymce.\n\n2. **Tùy chọn file HTML**: Mình thường thấy rằng file tinymce có thể được所在地 vào một base of data web là file `<html>` hoặc `<head>`. Tuy nhiên, base of data this sẽ không phải là base of data web usual, vì base of data web được sử dụng để lưu các phần tử HTML và CSS của tinymce.\n\n3. **Tùy chọn file CSS**: Base of data CSS có thể được所在地 vào một base of data web là file `<link>` hoặc `xml:stylesheet`. Tuy nhiên, base of data usual this sẽ không phải là base of data web usual, vì base of data web này không phải là base of data web usual.\n\nMожет bạn có cách nào để thay đổi base of data của bạn? Ví dụ, có thể đã có một base of data usual được sử dụng trong phần trình duyệt của bạn, nhưng base of data this sẽ không phải là base of data usual usual usual."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Đây là hình nhân vật gì ?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,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"
                    }
                }
            ]
        },
        {
            "role": "assistant",
            "content": "Đây là hình nhân vật gì là không!"
        },
        {
            "role": "user",
            "content": "xin chào"
        }
    ],
    "params": {},
    "features": {
        "image_generation": false,
        "web_search": false
    },
    "session_id": "y4HtT12jpwQh3g_IAAAB",
    "chat_id": "1c87b3b1-6970-44b3-b51e-2d999b2ac7c7",
    "id": "d08c6a96-1e6b-40d1-90e4-dcdf48ba0b7b"
}
'''
'''
response

{"status":true,"task_id":"4b45d073-9d14-4e52-bb8e-830cc70f6dca"}
'''
@app.post("/api/chat/completions") # Chat
async def chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
):
    if not request.app.state.MODELS:
        await get_all_models(request)

    tasks = form_data.pop("background_tasks", None)
    try:
        model_id = form_data.get("model", None)
        if model_id not in request.app.state.MODELS:
            raise Exception("Model not found")
        model = request.app.state.MODELS[model_id]

        # Check if user has access to the model
        if not BYPASS_MODEL_ACCESS_CONTROL and user.role == "user":
            try:
                check_model_access(user, model)
            except Exception as e:
                raise e

        metadata = {
            "user_id": user.id,
            "chat_id": form_data.pop("chat_id", None),
            "message_id": form_data.pop("id", None),
            "session_id": form_data.pop("session_id", None),
            "tool_ids": form_data.get("tool_ids", None),
            "files": form_data.get("files", None),
            "features": form_data.get("features", None),
        }
        form_data["metadata"] = metadata

        form_data, events = await process_chat_payload(
            request, form_data, metadata, user, model
        ) # functions, tools, files
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    try:
        response = await chat_completion_handler(request, form_data, user) # Chat
        return await process_chat_response(
            request, response, form_data, user, events, metadata, tasks
        ) # Chat
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# Alias for chat_completion (Legacy)
generate_chat_completions = chat_completion
generate_chat_completion = chat_completion


@app.post("/api/chat/completed")
async def chat_completed(
    request: Request, form_data: dict, user=Depends(get_verified_user)
):
    try:
        return await chat_completed_handler(request, form_data, user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post("/api/chat/actions/{action_id}")
async def chat_action(
    request: Request, action_id: str, form_data: dict, user=Depends(get_verified_user)
):
    try:
        return await chat_action_handler(request, action_id, form_data, user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post("/api/tasks/stop/{task_id}")
async def stop_task_endpoint(task_id: str, user=Depends(get_verified_user)):
    try:
        result = await stop_task(task_id)  # Use the function from tasks.py
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.get("/api/tasks")
async def list_tasks_endpoint(user=Depends(get_verified_user)):
    return {"tasks": list_tasks()}  # Use the function from tasks.py


##################################
#
# Config Endpoints
#
##################################


@app.get("/api/config")
async def get_app_config(request: Request):
    user = None
    if "token" in request.cookies:
        token = request.cookies.get("token")
        try:
            data = decode_token(token)
        except Exception as e:
            log.debug(e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        if data is not None and "id" in data:
            user = Users.get_user_by_id(data["id"])

    onboarding = False
    if user is None:
        user_count = Users.get_num_users()
        onboarding = user_count == 0

    return {
        **({"onboarding": True} if onboarding else {}),
        "status": True,
        "name": WEBUI_NAME,
        "version": VERSION,
        "default_locale": str(DEFAULT_LOCALE),
        "oauth": {
            "providers": {
                name: config.get("name", name)
                for name, config in OAUTH_PROVIDERS.items()
            }
        },
        "features": {
            "auth": WEBUI_AUTH,
            "auth_trusted_header": bool(app.state.AUTH_TRUSTED_EMAIL_HEADER),
            "enable_ldap": app.state.config.ENABLE_LDAP,
            "enable_api_key": app.state.config.ENABLE_API_KEY,
            "enable_signup": app.state.config.ENABLE_SIGNUP,
            "enable_login_form": app.state.config.ENABLE_LOGIN_FORM,
            "enable_websocket": ENABLE_WEBSOCKET_SUPPORT,
            **(
                {
                    "enable_channels": app.state.config.ENABLE_CHANNELS,
                    "enable_web_search": app.state.config.ENABLE_RAG_WEB_SEARCH,
                    "enable_google_drive_integration": app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION,
                    "enable_image_generation": app.state.config.ENABLE_IMAGE_GENERATION,
                    "enable_community_sharing": app.state.config.ENABLE_COMMUNITY_SHARING,
                    "enable_message_rating": app.state.config.ENABLE_MESSAGE_RATING,
                    "enable_admin_export": ENABLE_ADMIN_EXPORT,
                    "enable_admin_chat_access": ENABLE_ADMIN_CHAT_ACCESS,
                }
                if user is not None
                else {}
            ),
        },
        "google_drive": {
            "client_id": GOOGLE_DRIVE_CLIENT_ID.value,
            "api_key": GOOGLE_DRIVE_API_KEY.value,
        },
        **(
            {
                "default_models": app.state.config.DEFAULT_MODELS,
                "default_prompt_suggestions": app.state.config.DEFAULT_PROMPT_SUGGESTIONS,
                "audio": {
                    "tts": {
                        "engine": app.state.config.TTS_ENGINE,
                        "voice": app.state.config.TTS_VOICE,
                        "split_on": app.state.config.TTS_SPLIT_ON,
                    },
                    "stt": {
                        "engine": app.state.config.STT_ENGINE,
                    },
                },
                "file": {
                    "max_size": app.state.config.FILE_MAX_SIZE,
                    "max_count": app.state.config.FILE_MAX_COUNT,
                },
                "permissions": {**app.state.config.USER_PERMISSIONS},
            }
            if user is not None
            else {}
        ),
    }


class UrlForm(BaseModel):
    url: str


@app.get("/api/webhook")
async def get_webhook_url(user=Depends(get_admin_user)):
    return {
        "url": app.state.config.WEBHOOK_URL,
    }


@app.post("/api/webhook")
async def update_webhook_url(form_data: UrlForm, user=Depends(get_admin_user)):
    app.state.config.WEBHOOK_URL = form_data.url
    app.state.WEBHOOK_URL = app.state.config.WEBHOOK_URL
    return {"url": app.state.config.WEBHOOK_URL}


@app.get("/api/version")
async def get_app_version():
    return {
        "version": VERSION,
    }


@app.get("/api/version/updates")
async def get_app_latest_release_version():
    if OFFLINE_MODE:
        log.debug(
            f"Offline mode is enabled, returning current version as latest version"
        )
        return {"current": VERSION, "latest": VERSION}
    try:
        timeout = aiohttp.ClientTimeout(total=1)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.get(
                "https://api.github.com/repos/open-webui/open-webui/releases/latest"
            ) as response:
                response.raise_for_status()
                data = await response.json()
                latest_version = data["tag_name"]

                return {"current": VERSION, "latest": latest_version[1:]}
    except Exception as e:
        log.debug(e)
        return {"current": VERSION, "latest": VERSION}


@app.get("/api/changelog")
async def get_app_changelog():
    return {key: CHANGELOG[key] for idx, key in enumerate(CHANGELOG) if idx < 5}


############################
# OAuth Login & Callback
############################

# SessionMiddleware is used by authlib for oauth
if len(OAUTH_PROVIDERS) > 0:
    app.add_middleware(
        SessionMiddleware,
        secret_key=WEBUI_SECRET_KEY,
        session_cookie="oui-session",
        same_site=WEBUI_SESSION_COOKIE_SAME_SITE,
        https_only=WEBUI_SESSION_COOKIE_SECURE,
    )


@app.get("/oauth/{provider}/login")
async def oauth_login(provider: str, request: Request):
    return await oauth_manager.handle_login(provider, request)


# OAuth login logic is as follows:
# 1. Attempt to find a user with matching subject ID, tied to the provider
# 2. If OAUTH_MERGE_ACCOUNTS_BY_EMAIL is true, find a user with the email address provided via OAuth
#    - This is considered insecure in general, as OAuth providers do not always verify email addresses
# 3. If there is no user, and ENABLE_OAUTH_SIGNUP is true, create a user
#    - Email addresses are considered unique, so we fail registration if the email address is already taken
@app.get("/oauth/{provider}/callback")
async def oauth_callback(provider: str, request: Request, response: Response):
    return await oauth_manager.handle_callback(provider, request, response)


@app.get("/manifest.json")
async def get_manifest_json():
    return {
        "name": WEBUI_NAME,
        "short_name": WEBUI_NAME,
        "description": "Open WebUI is an open, extensible, user-friendly interface for AI that adapts to your workflow.",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#343541",
        "orientation": "natural",
        "icons": [
            {
                "src": "/static/logo.png",
                "type": "image/png",
                "sizes": "500x500",
                "purpose": "any",
            },
            {
                "src": "/static/logo.png",
                "type": "image/png",
                "sizes": "500x500",
                "purpose": "maskable",
            },
        ],
    }


@app.get("/opensearch.xml")
async def get_opensearch_xml():
    xml_content = rf"""
    <OpenSearchDescription xmlns="http://a9.com/-/spec/opensearch/1.1/" xmlns:moz="http://www.mozilla.org/2006/browser/search/">
    <ShortName>{WEBUI_NAME}</ShortName>
    <Description>Search {WEBUI_NAME}</Description>
    <InputEncoding>UTF-8</InputEncoding>
    <Image width="16" height="16" type="image/x-icon">{app.state.config.WEBUI_URL}/static/favicon.png</Image>
    <Url type="text/html" method="get" template="{app.state.config.WEBUI_URL}/?q={"{searchTerms}"}"/>
    <moz:SearchForm>{app.state.config.WEBUI_URL}</moz:SearchForm>
    </OpenSearchDescription>
    """
    return Response(content=xml_content, media_type="application/xml")


@app.get("/health")
async def healthcheck():
    return {"status": True}


@app.get("/health/db")
async def healthcheck_with_db():
    Session.execute(text("SELECT 1;")).all()
    return {"status": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/cache", StaticFiles(directory=CACHE_DIR), name="cache")


def swagger_ui_html(*args, **kwargs):
    return get_swagger_ui_html(
        *args,
        **kwargs,
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        swagger_favicon_url="/static/swagger-ui/favicon.png",
    )


applications.get_swagger_ui_html = swagger_ui_html

if os.path.exists(FRONTEND_BUILD_DIR):
    mimetypes.add_type("text/javascript", ".js")
    app.mount(
        "/",
        SPAStaticFiles(directory=FRONTEND_BUILD_DIR, html=True),
        name="spa-static-files",
    )
else:
    log.warning(
        f"Frontend build directory not found at '{FRONTEND_BUILD_DIR}'. Serving API only."
    )
