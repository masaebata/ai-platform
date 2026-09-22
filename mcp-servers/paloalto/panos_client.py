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