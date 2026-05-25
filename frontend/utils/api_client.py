"""
Client API pour communiquer avec FastAPI backend.
"""

import requests
from typing import Optional, Dict, Any

class APIClient:
    """Client pour l'API Diffusion Language Model"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.timeout = 120
    
    def health_check(self) -> Dict[str, Any]:
        """Vérifier l'état de l'API"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate(
        self,
        prompt: str,
        steps: int = 30,
        temperature: float = 0.8,
        max_length: int = 100,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Générer du texte via l'API"""
        try:
            payload = {
                "prompt": prompt,
                "steps": steps,
                "temperature": temperature,
                "max_length": max_length,
                "verbose": verbose
            }
            
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Connexion refusée - API non démarrée"}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Timeout - génération trop longue"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def batch_generate(
        self,
        prompts: list,
        steps: int = 30,
        temperature: float = 0.8,
        max_length: int = 100
    ) -> Dict[str, Any]:
        """Génération par lots"""
        try:
            payload = {
                "prompts": prompts,
                "steps": steps,
                "temperature": temperature,
                "max_length": max_length
            }
            
            response = requests.post(
                f"{self.base_url}/batch_generate",
                json=payload,
                timeout=self.timeout * len(prompts)
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}


def get_api_client() -> APIClient:
    """Factory pour obtenir le client API"""
    import streamlit as st
    
    # Essayer de lire l'URL depuis les secrets Streamlit
    try:
        api_url = st.secrets.get("API_URL", "http://localhost:8000")
    except:
        api_url = "http://localhost:8000"
    
    return APIClient(base_url=api_url)