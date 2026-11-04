"""Implementaciones de ``BaseLLMClient`` para cada proveedor.

Es el único lugar del proyecto donde se importan los SDKs ``openai`` y
``anthropic`` y sus excepciones. Fuera de este paquete, el código usa solo
nuestros schemas y nuestras excepciones.
"""
