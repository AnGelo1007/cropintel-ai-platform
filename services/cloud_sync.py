from __future__ import annotations
import os
from pathlib import Path
from typing import Any

try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None
    Client = object


class OptionalSupabaseSync:
    def __init__(self, enabled: bool, client: Any = None, bucket: str = 'crop-images'):
        self.enabled = enabled
        self.client = client
        self.bucket = bucket

    @classmethod
    def from_env(cls) -> 'OptionalSupabaseSync':
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        bucket = os.getenv('SUPABASE_BUCKET', 'crop-images')
        if not url or not key or create_client is None:
            return cls(False)
        client = create_client(url, key)
        return cls(True, client=client, bucket=bucket)

    def try_sync(self, result: dict[str, Any], image_path: Path) -> dict[str, Any]:
        if not self.enabled or self.client is None:
            return {'enabled': False, 'message': 'Supabase sync disabled.'}
        try:
            storage_path = f"uploads/{image_path.name}"
            with open(image_path, 'rb') as f:
                self.client.storage.from_(self.bucket).upload(
                    path=storage_path,
                    file=f,
                    file_options={'content-type': 'image/jpeg', 'upsert': 'true'}
                )
            self.client.table('crop_records').insert({
                'captured_at': result['captured_at'],
                'crop_name': result['crop_name'],
                'crop_confidence': result['crop_confidence'],
                'health_status': result['health_status'],
                'health_score': result['health_score'],
                'disease_risk': result['disease_risk'],
                'yield_forecast_kg': result['yield_forecast_kg'],
                'image_path': storage_path,
                'notes': result['notes'],
            }).execute()
            return {'enabled': True, 'message': 'Synced to Supabase.'}
        except Exception as exc:
            return {'enabled': True, 'message': f'Supabase sync failed: {exc}'}
