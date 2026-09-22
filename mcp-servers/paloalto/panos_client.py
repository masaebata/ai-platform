cd /opt/ai-app/actions-runner/_work/ai-platform/ai-platform

sudo docker compose --env-file /etc/ai-platform/.env config >/dev/null && echo "Compose OK"
 
sudo docker compose --env-file /etc/ai-platform/.env build --no-cache ai-agent
 
sudo docker compose --env-file /etc/ai-platform/.env up -d --force-recreate ai-agent open-webui
 
sudo docker exec ai-agent python -c 'import app; print(app.app.version); print([(r.path, sorted(r.methods or [])) for r in app.app.routes])'
 
set -a; source <(sudo cat /etc/ai-platform/.env); set +a
 
curl http://127.0.0.1:8081/v1/models -H "Authorization: Bearer $AGENT_API_KEY"
 
sudo docker exec ai-open-webui sh -c 'echo "$OPENAI_API_BASE_URLS"'
 
sudo docker exec ai-open-webui sh -c 'echo "$OPENAI_API_KEYS" | sed "s/[^;]*/***/g"'
 
sudo docker exec ai-open-webui python -c "import os,requests; urls=os.environ['OPENAI_API_BASE_URLS'].split(';'); keys=os.environ['OPENAI_API_KEYS'].split(';'); r=requests.get(urls[1]+'/models',headers={'Authorization':'Bearer '+keys[1]}); print(r.status_code); print(r.text)"
 
cd /opt/ai-app/actions-runner/_work/ai-platform/ai-platform
 
sudo docker compose --env-file /etc/ai-platform/.env up -d --force-recreate open-webui
 
sudo docker ps | grep ai-open-webui
 
services:
 
  postgres:

    image: postgres:16

    container_name: ai-postgres

    restart: unless-stopped
 
    environment:

      POSTGRES_USER: ${POSTGRES_USER}

      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}

      POSTGRES_DB: ${POSTGRES_DB}
 
    volumes:

      - postgres-data:/var/lib/postgresql/data
 
    networks:

      - ai-network
 
    healthcheck:

      test:

        - CMD-SHELL

        - pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}

      interval: 10s

      timeout: 5s

      retries: 5
 
 
  qdrant:

    image: qdrant/qdrant:latest

    container_name: ai-qdrant

    restart: unless-stopped
 
    ports:

      - "6333:6333"
 
    volumes:

      - qdrant-data:/qdrant/storage
 
    networks:

      - ai-network
 
 
  litellm:

    image: docker.litellm.ai/berriai/litellm:main-latest

    container_name: ai-litellm

    restart: unless-stopped
 
    ports:

      - "4000:4000"
 
    environment:

      LITELLM_MASTER_KEY: ${LITELLM_MASTER_KEY}
 
    volumes:

      - ./litellm/config.yaml:/app/config.yaml:ro
 
    command:

      - "--config"

      - "/app/config.yaml"

      - "--port"

      - "4000"
 
    networks:

      - ai-network
 
 
  paloalto-mcp:

    build:

      context: ./mcp-servers/paloalto
 
    container_name: ai-paloalto-mcp

    restart: unless-stopped
 
    ports:

      - "8000:8000"
 
    environment:

      PANOS_HOST: ${PANOS_HOST}

      PANOS_API_KEY: ${PANOS_API_KEY}

      PANOS_VERIFY_SSL: ${PANOS_VERIFY_SSL:-false}

      PANOS_TIMEOUT: ${PANOS_TIMEOUT:-30}
 
    networks:

      - ai-network
 
 
  ai-agent:

    build:

      context: ./agent
 
    container_name: ai-agent

    restart: unless-stopped
 
    ports:

      - "8081:8080"
 
    environment:

      OLLAMA_BASE_URL: http://192.168.11.129:11434/v1

      OLLAMA_MODEL: qwen3:4b
 
      AGENT_MODEL_ID: paloalto-agent

      AGENT_API_KEY: ${AGENT_API_KEY}

      AGENT_TIMEOUT: ${AGENT_TIMEOUT:-180}
 
      PALOALTO_MCP_URL: http://paloalto-mcp:8000/mcp
 
    depends_on:

      - paloalto-mcp
 
    networks:

      - ai-network
 
 
  open-webui:

    image: ghcr.io/open-webui/open-webui:main

    container_name: ai-open-webui

    restart: unless-stopped
 
    ports:

      - "3000:8080"
 
    environment:

      WEBUI_SECRET_KEY: ${WEBUI_SECRET_KEY}
 
      ENABLE_OPENAI_API: "true"
 
      OPENAI_API_BASE_URLS: "http://litellm:4000/v1;http://ai-agent:8080/v1"

      OPENAI_API_KEYS: "${LITELLM_MASTER_KEY};${AGENT_API_KEY}"
 
      ENABLE_PERSISTENT_CONFIG: "false"

      RESET_CONFIG_ON_START: "true"
 
    volumes:

      - open-webui-data:/app/backend/data
 
    depends_on:

      - litellm

      - ai-agent
 
    networks:

      - ai-network
 
 
  mcp-client:

    image: python:3.12-slim

    container_name: ai-mcp-client
 
    working_dir: /app
 
    volumes:

      - ./mcp-client:/app:ro
 
    command:

      - sh

      - -c

      - "pip install --no-cache-dir -r requirements.txt && python test_paloalto_mcp.py"
 
    depends_on:

      - paloalto-mcp
 
    networks:

      - ai-network
 
    profiles:

      - test
 
 
networks:

  ai-network:

    driver: bridge
 
 
volumes:

  postgres-data:

  qdrant-data:

  open-webui-data:
 
git add docker-compose.yml

git commit -m "Force Open WebUI to use configured AI connections"

git push origin main
 
cd /opt/ai-app/actions-runner/_work/ai-platform/ai-platform
 
sudo docker compose --env-file /etc/ai-platform/.env config >/dev/null && echo "Compose OK"
 
sudo docker compose --env-file /etc/ai-platform/.env up -d --force-recreate open-webui
 
import os

from typing import Any
 
import httpx

import xmltodict
 
 
class PanOSClient:

    def __init__(self) -> None:

        self.host = os.environ["PANOS_HOST"].rstrip("/")

        self.api_key = os.environ["PANOS_API_KEY"]
 
        verify_env = os.getenv(

            "PANOS_VERIFY_SSL",

            "false",

        ).lower()
 
        self.verify_ssl = verify_env in (

            "true",

            "1",

            "yes",

        )
 
        self.timeout = float(

            os.getenv(

                "PANOS_TIMEOUT",

                "30",

            )

        )
 
    async def _request(

        self,

        request_type: str,

        **params: str,

    ) -> dict[str, Any]:
 
        url = f"{self.host}/api/"
 
        query = {

            "type": request_type,

            "key": self.api_key,

            **params,

        }
 
        async with httpx.AsyncClient(

            verify=self.verify_ssl,

            timeout=self.timeout,

        ) as client:
 
            response = await client.post(

                url,

                params=query,

            )
 
        response.raise_for_status()
 
        data = xmltodict.parse(

            response.text

        )
 
        api_response = data.get(

            "response",

            {},

        )
 
        status = api_response.get(

            "@status"

        )
 
        if status != "success":
 
            msg = api_response.get(

                "msg",

                "Unknown PAN-OS API error",

            )
 
            raise RuntimeError(

                f"PAN-OS API error: {msg}"

            )
 
        return api_response
 
    async def op(

        self,

        cmd: str,

    ) -> dict[str, Any]:
 
        response = await self._request(

            "op",

            cmd=cmd,

        )
 
        return response.get(

            "result",

            {},

        )
 
    async def get_device_info(

        self,

    ) -> dict[str, Any]:
 
        cmd = """
<show>
<system>
<info/>
</system>
</show>

        """
 
        result = await self.op(

            cmd

        )
 
        system = result.get(

            "system",

            result,

        )
 
        return {

            "hostname": system.get(

                "hostname"

            ),

            "ip_address": system.get(

                "ip-address"

            ),

            "netmask": system.get(

                "netmask"

            ),

            "default_gateway": system.get(

                "default-gateway"

            ),

            "model": system.get(

                "model"

            ),

            "serial": system.get(

                "serial"

            ),

            "sw_version": system.get(

                "sw-version"

            ),

            "app_version": system.get(

                "app-version"

            ),

            "av_version": system.get(

                "av-version"

            ),

            "wildfire_version": system.get(

                "wildfire-version"

            ),

            "uptime": system.get(

                "uptime"

            ),

            "family": system.get(

                "family"

            ),

        }
 
    async def get_interfaces(

        self,

    ) -> dict[str, Any]:
 
        cmd = """
<show>
<interface>all</interface>
</show>

        """
 
        result = await self.op(

            cmd

        )
 
        return result
 
from fastmcp import FastMCP
 
from panos_client import PanOSClient
 
 
mcp = FastMCP(

    name="Palo Alto Networks MCP Server"

)
 
 
@mcp.tool

async def get_device_info() -> dict:

    """

    Get basic system information from the configured

    Palo Alto Networks firewall.
 
    Read-only operation.

    """
 
    client = PanOSClient()
 
    return await client.get_device_info()
 
 
@mcp.tool

async def get_interfaces() -> dict:

    """

    Get all interface information from the configured

    Palo Alto Networks firewall.
 
    Equivalent to:

    show interface all
 
    Read-only operation.

    """
 
    client = PanOSClient()
 
    return await client.get_interfaces()
 
 
if __name__ == "__main__":
 
    mcp.run(

        transport="http",

        host="0.0.0.0",

        port=8000,

    )
 
git add mcp-servers\paloalto\panos_client.py mcp-servers\paloalto\server.py

git commit -m "Add Palo Alto interface MCP tool"

git push origin main
 
sudo docker compose --env-file /etc/ai-platform/.env --profile test run --rm mcp-client
 
sudo docker restart ai-agent
 
curl http://127.0.0.1:8081/health
 
curl -X POST http://127.0.0.1:8081/chat -H "Content-Type: application/json" -d '{"message":"PA-VMのインターフェース一覧と状態を確認してください。"}'
 
sudo systemctl status ollama
 
ollama ps

ollama list
 
curl http://192.168.11.129:11434/api/tags
 
curl http://192.168.11.129:11434/api/tags
 
time curl --max-time 600 -X POST http://127.0.0.1:8081/chat -H "Content-Type: application/json" -d '{"message":"PA-VMの機器情報を確認して、日本語で要約してください。"}'
 
import os

from typing import Any
 
import httpx

import xmltodict
 
 
class PanOSClient:

    def __init__(self) -> None:

        self.host = os.environ["PANOS_HOST"].rstrip("/")

        self.api_key = os.environ["PANOS_API_KEY"]
 
        verify_env = os.getenv(

            "PANOS_VERIFY_SSL",

            "false",

        ).lower()
 
        self.verify_ssl = verify_env in (

            "true",

            "1",

            "yes",

        )
 
        self.timeout = float(

            os.getenv(

                "PANOS_TIMEOUT",

                "30",

            )

        )
 
        self.vsys = os.getenv(

            "PANOS_VSYS",

            "vsys1",

        )
 
    async def _request(

        self,

        request_type: str,

        **params: str,

    ) -> dict[str, Any]:
 
        url = f"{self.host}/api/"
 
        query = {

            "type": request_type,

            "key": self.api_key,

            **params,

        }
 
        async with httpx.AsyncClient(

            verify=self.verify_ssl,

            timeout=self.timeout,

        ) as client:
 
            response = await client.post(

                url,

                params=query,

            )
 
        response.raise_for_status()
 
        data = xmltodict.parse(

            response.text

        )
 
        api_response = data.get(

            "response",

            {},

        )
 
        status = api_response.get(

            "@status"

        )
 
        if status != "success":
 
            msg = api_response.get(

                "msg",

                "Unknown PAN-OS API error",

            )
 
            raise RuntimeError(

                f"PAN-OS API error: {msg}"

            )
 
        return api_response
 
    async def op(

        self,

        cmd: str,

    ) -> dict[str, Any]:
 
        response = await self._request(

            "op",

            cmd=cmd,

        )
 
        return response.get(

            "result",

            {},

        )
 
    async def config_show(

        self,

        xpath: str,

    ) -> dict[str, Any]:
 
        response = await self._request(

            "config",

            action="show",

            xpath=xpath,

        )
 
        return response.get(

            "result",

            {},

        )
 
    async def get_device_info(

        self,

    ) -> dict[str, Any]:
 
        cmd = """
<show>
<system>
<info/>
</system>
</show>

        """
 
        result = await self.op(

            cmd

        )
 
        system = result.get(

            "system",

            result,

        )
 
        return {

            "hostname": system.get(

                "hostname"

            ),

            "ip_address": system.get(

                "ip-address"

            ),

            "netmask": system.get(

                "netmask"

            ),

            "default_gateway": system.get(

                "default-gateway"

            ),

            "model": system.get(

                "model"

            ),

            "serial": system.get(

                "serial"

            ),

            "sw_version": system.get(

                "sw-version"

            ),

            "app_version": system.get(

                "app-version"

            ),

            "av_version": system.get(

                "av-version"

            ),

            "wildfire_version": system.get(

                "wildfire-version"

            ),

            "uptime": system.get(

                "uptime"

            ),

            "family": system.get(

                "family"

            ),

        }
 
    async def get_interfaces(

        self,

    ) -> dict[str, Any]:
 
        cmd = """
<show>
<interface>all</interface>
</show>

        """
 
        return await self.op(

            cmd

        )
 
    async def get_routes(

        self,

    ) -> dict[str, Any]:
 
        cmd = """
<show>
<routing>
<route/>
</routing>
</show>

        """
 
        return await self.op(

            cmd

        )
 
    async def get_security_rules(

        self,

    ) -> dict[str, Any]:
 
        xpath = (

            "/config/devices/entry"

            f"/vsys/entry[@name='{self.vsys}']"

            "/rulebase/security/rules"

        )
 
        result = await self.config_show(

            xpath

        )
 
        return {

            "vsys": self.vsys,

            "rules": result,

        }
 