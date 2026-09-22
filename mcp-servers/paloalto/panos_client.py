import asyncio
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
 
        result = await self.op(cmd)
 
        system = result.get(
            "system",
            result,
        )
 
        return {
            "hostname": system.get("hostname"),
            "ip_address": system.get("ip-address"),
            "netmask": system.get("netmask"),
            "default_gateway": system.get("default-gateway"),
            "model": system.get("model"),
            "serial": system.get("serial"),
            "sw_version": system.get("sw-version"),
            "app_version": system.get("app-version"),
            "av_version": system.get("av-version"),
            "wildfire_version": system.get("wildfire-version"),
            "uptime": system.get("uptime"),
            "family": system.get("family"),
        }
 
    async def get_interfaces(
        self,
    ) -> dict[str, Any]:
 
        cmd = """
<show>
<interface>all</interface>
</show>
        """
 
        return await self.op(cmd)
 
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
 
        return await self.op(cmd)
 
    async def get_security_rules(
        self,
    ) -> dict[str, Any]:
 
        xpath = (
            "/config/devices/entry"
            f"/vsys/entry[@name='{self.vsys}']"
            "/rulebase/security/rules"
        )
 
        result = await self.config_show(xpath)
 
        rules = (
            result
            .get("rules", {})
            .get("entry", [])
        )
 
        if isinstance(rules, dict):
            rules = [rules]
 
        normalized_rules = []
 
        for rule in rules:
            normalized_rules.append(
                {
                    "name": rule.get("@name"),
                    "from": (
                        rule.get("from", {})
                        .get("member", [])
                    ),
                    "to": (
                        rule.get("to", {})
                        .get("member", [])
                    ),
                    "source": (
                        rule.get("source", {})
                        .get("member", [])
                    ),
                    "destination": (
                        rule.get("destination", {})
                        .get("member", [])
                    ),
                    "application": (
                        rule.get("application", {})
                        .get("member", [])
                    ),
                    "service": (
                        rule.get("service", {})
                        .get("member", [])
                    ),
                    "action": rule.get("action"),
                    "log_start": rule.get("log-start"),
                    "log_end": rule.get("log-end"),
                    "disabled": rule.get("disabled"),
                    "description": rule.get("description"),
                }
            )
 
        return {
            "vsys": self.vsys,
            "count": len(normalized_rules),
            "rules": normalized_rules,
        }
 
    async def get_threat_logs(
        self,
        nlogs: int = 20,
        query: str | None = None,
    ) -> dict[str, Any]:
 
        params = {
            "log-type": "threat",
            "nlogs": str(nlogs),
            "dir": "backward",
        }
 
        if query:
            params["query"] = query
 
        initial = await self._request(
            "log",
            **params,
        )
 
        result = initial.get(
            "result",
            {},
        )
 
        job_id = result.get("job")
 
        if not job_id:
            raise RuntimeError(
                "PAN-OS Threat Log job ID was not returned"
            )
 
        for _ in range(30):
 
            await asyncio.sleep(2)
 
            response = await self._request(
                "log",
                action="get",
                **{
                    "job-id": str(job_id)
                },
            )
 
            result = response.get(
                "result",
                {},
            )
 
            job = result.get(
                "job",
                {},
            )
 
            status = None
 
            if isinstance(job, dict):
                status = job.get("status")
 
            log_node = result.get("log")
 
            if log_node:
                logs = (
                    log_node
                    .get("logs", {})
                    .get("entry", [])
                )
 
                if isinstance(logs, dict):
                    logs = [logs]
 
                normalized_logs = []
 
                for entry in logs:
                    normalized_logs.append(
                        {
                            "receive_time": entry.get(
                                "receive_time"
                            ),
                            "src": entry.get("src"),
                            "dst": entry.get("dst"),
                            "rule": entry.get("rule"),
                            "application": entry.get("app"),
                            "source_zone": entry.get("from"),
                            "destination_zone": entry.get("to"),
                            "source_port": entry.get("sport"),
                            "destination_port": entry.get("dport"),
                            "protocol": entry.get("proto"),
                            "action": entry.get("action"),
                            "severity": entry.get("severity"),
                            "subtype": entry.get("subtype"),
                            "threat_id": entry.get("threatid"),
                            "category": entry.get("category"),
                            "url_filename": entry.get("url"),
                        }
                    )
 
                return {
                    "count": len(normalized_logs),
                    "logs": normalized_logs,
                }
 
            if status == "FIN":
                return {
                    "count": 0,
                    "logs": [],
                }
 
        raise TimeoutError(
            "Timed out waiting for PAN-OS Threat Log job"
        )