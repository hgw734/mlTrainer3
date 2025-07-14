#!/usr/bin/env python3
"""
Mandatory Execution Validator - Forces all code to be executed in isolation before acceptance
No code can be deployed without proof of successful execution
"""

import subprocess
import tempfile
import ast
import os
import hashlib
from pathlib import Path
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import docker
import resource
import signal
from contextlib import contextmanager
import logging
import io

# Import immutable rules
from .immutable_rules_kernel import IMMUTABLE_RULES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VALIDATOR - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console only for now
    ]
)
logger = logging.getLogger(__name__)

class ComplianceViolation(Exception):
    """Exception raised when code violates compliance rules"""
    pass

class MandatoryExecutionValidator:
    """
    Forces all code to be executed in isolation before acceptance
    """
    
    def __init__(self):
        self.validation_cache = {}
        self.docker_available = self._check_docker()
        self.validation_image = "mltrainer/validation:immutable"
        
    def _check_docker(self) -> bool:
        """Check if Docker is available for isolated execution"""
        try:
            import docker
            client = docker.from_env()
            client.ping()
            return True
        except:
            logger.warning("Docker not available - using subprocess isolation")
            return False
    
    def validate_code(self, code_path: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Validate code by actually running it
        Returns validation certificate or raises exception
        """
        code_path = Path(code_path)
        if not code_path.exists():
            raise FileNotFoundError(f"Code file not found: {code_path}")
        
        # Check cache first
        file_hash = self._calculate_file_hash(code_path)
        if file_hash in self.validation_cache:
            cached = self.validation_cache[file_hash]
            if time.time() - cached['timestamp'] < 3600:  # 1 hour cache
                logger.info(f"Using cached validation for {code_path}")
                return cached['certificate']
        
        logger.info(f"Validating code: {code_path}")
        
        # Step 1: Static analysis
        static_result = self._static_analysis(code_path)
        if not static_result["passed"]:
            raise ComplianceViolation(f"Static analysis failed: {static_result['errors']}")
        
        # Step 2: Isolated execution
        execution_result = self._isolated_execution(code_path, timeout)
        if not execution_result["passed"]:
            raise ComplianceViolation(f"Execution failed: {execution_result['errors']}")
        
        # Step 3: Runtime behavior analysis
        behavior_result = self._analyze_runtime_behavior(execution_result)
        if not behavior_result["passed"]:
            raise ComplianceViolation(f"Behavior analysis failed: {behavior_result['errors']}")
        
        # Generate immutable certificate
        certificate = self._generate_certificate(code_path, static_result, execution_result, behavior_result)
        
        # Cache the result
        self.validation_cache[file_hash] = {
            'certificate': certificate,
            'timestamp': time.time()
        }
        
        return certificate
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _static_analysis(self, code_path: Path) -> Dict[str, Any]:
        """Deep static analysis to catch deceptive patterns"""
        with open(code_path, 'r') as f:
            code = f.read()
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"passed": False, "errors": [f"Syntax error: {e}"]}
        
        errors = []
        warnings = []
        
        # Check for deceptive patterns
        for node in ast.walk(tree):
            # Pattern: function().method() where function might return None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Call):
                    # This is a chained call - verify it's legitimate
                    call_chain = self._extract_call_chain(node)
                    if self._is_suspicious_chain(call_chain):
                        errors.append(
                            f"Suspicious call chain: {'.'.join(call_chain)} at line {node.lineno}"
                        )
            
            # Pattern: Import from non-existent module
            if isinstance(node, ast.ImportFrom):
                if node.module == "ml_engine_real":
                    # Check specific imports
                    for alias in node.names:
                        if alias.name == "get_market_data":
                            errors.append(
                                f"Deceptive import detected: 'get_market_data' is not a module-level function "
                                f"in ml_engine_real at line {node.lineno}"
                            )
            
            # Pattern: Prohibited patterns in strings
            if isinstance(node, ast.Str):
                for pattern in IMMUTABLE_RULES.get_rule("prohibited_patterns"):
                    if pattern in node.s:
                        warnings.append(f"Prohibited pattern '{pattern}' in string at line {node.lineno}")
        
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "ast_nodes": len(list(ast.walk(tree)))
        }
    
    def _extract_call_chain(self, node: ast.Call) -> List[str]:
        """Extract call chain from AST node"""
        chain = []
        current = node
        
        while current:
            if isinstance(current, ast.Call):
                if isinstance(current.func, ast.Attribute):
                    chain.insert(0, current.func.attr)
                    current = current.func.value
                elif isinstance(current.func, ast.Name):
                    chain.insert(0, current.func.id)
                    break
                else:
                    break
            elif isinstance(current, ast.Name):
                chain.insert(0, current.id)
                break
            else:
                break
        
        return chain
    
    def _is_suspicious_chain(self, chain: List[str]) -> bool:
        """Check if call chain looks suspicious"""
        # Known problematic patterns
        if len(chain) >= 2:
            if chain[-1] in ["get_volatility", "sample_historical"]:
                # These methods don't exist in the codebase
                return True
            if chain[0] == "get_market_data" and len(chain) > 1:
                # get_market_data returns a DataFrame, not an object with these methods
                return True
        return False
    
    def _isolated_execution(self, code_path: Path, timeout: int) -> Dict[str, Any]:
        """Execute code in completely isolated environment"""
        if self.docker_available:
            return self._docker_execution(code_path, timeout)
        else:
            return self._subprocess_execution(code_path, timeout)
    
    def _subprocess_execution(self, code_path: Path, timeout: int) -> Dict[str, Any]:
        """Execute in subprocess with resource limits"""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy code to temp directory
            tmp_code = Path(tmpdir) / code_path.name
            tmp_code.write_text(code_path.read_text())
            
            # Create execution wrapper
            wrapper = Path(tmpdir) / "wrapper.py"
            wrapper_code = f"""
import sys
import traceback
import json

# Add enforcement hooks
sys.path.insert(0, '{Path(__file__).parent}')
from runtime_enforcement_hooks import ENFORCEMENT_HOOKS

try:
    # Execute the code
    exec(open('{tmp_code.name}').read())
    result = {{"success": True, "output": "Code executed successfully"}}
except Exception as e:
    result = {{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}

print(json.dumps(result))
"""
            wrapper.write_text(wrapper_code)
            
            # Set resource limits
            def set_limits():
                # CPU time limit
                resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
                # Memory limit (256MB)
                resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
                # No network access (would need additional sandboxing)
            
            # Execute with timeout
            try:
                proc = subprocess.Popen(
                    [sys.executable, str(wrapper)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=tmpdir,
                    preexec_fn=set_limits,
                    env={**os.environ, 'PYTHONPATH': str(Path(__file__).parent.parent)}
                )
                
                stdout, stderr = proc.communicate(timeout=timeout)
                
                # Parse result
                try:
                    result = json.loads(stdout.decode())
                except:
                    result = {"success": False, "error": "Failed to parse output"}
                
                # Check for specific error patterns
                errors = []
                stderr_text = stderr.decode()
                
                if "ImportError" in stderr_text:
                    errors.append(f"Import error detected: {stderr_text}")
                if "AttributeError" in stderr_text and "get_volatility" in stderr_text:
                    errors.append(f"Fake method call detected: {stderr_text}")
                if "'NoneType' object has no attribute" in stderr_text:
                    errors.append(f"None type method call: {stderr_text}")
                
                return {
                    "passed": proc.returncode == 0 and result.get("success", False) and len(errors) == 0,
                    "errors": errors + ([result.get("error", "")] if not result.get("success") else []),
                    "exit_code": proc.returncode,
                    "output": result,
                    "stderr": stderr_text
                }
                
            except subprocess.TimeoutExpired:
                proc.kill()
                return {
                    "passed": False,
                    "errors": [f"Execution timeout ({timeout}s)"],
                    "exit_code": -1
                }
            except Exception as e:
                return {
                    "passed": False,
                    "errors": [f"Execution failed: {str(e)}"],
                    "exit_code": -1
                }
    
    def _docker_execution(self, code_path: Path, timeout: int) -> Dict[str, Any]:
        """Execute code in Docker container with strict limits"""
        import docker
        client = docker.from_env()
        
        # Ensure validation image exists
        try:
            client.images.get(self.validation_image)
        except docker.errors.ImageNotFound:
            # Build the image
            self._build_validation_image(client)
        
        # Create container with strict limits
        container = client.containers.run(
            self.validation_image,
            command=f"python /code/{code_path.name}",
            volumes={
                str(code_path.parent): {"bind": "/code", "mode": "ro"},
                str(Path(__file__).parent): {"bind": "/enforcement", "mode": "ro"}
            },
            mem_limit="256m",
            cpu_quota=50000,  # 50% of one CPU
            network_mode="none",  # No network access
            security_opt=["no-new-privileges"],
            detach=True,
            stdout=True,
            stderr=True,
            environment={
                "PYTHONPATH": "/enforcement:/code"
            }
        )
        
        # Wait for completion with timeout
        try:
            result = container.wait(timeout=timeout)
            stdout = container.logs(stdout=True, stderr=False).decode()
            stderr = container.logs(stdout=False, stderr=True).decode()
            
            # Check for specific error patterns
            errors = []
            if "ImportError" in stderr:
                if "get_market_data" in stderr:
                    errors.append("Deceptive import: get_market_data is not a module-level function")
                else:
                    errors.append(f"Import error: {stderr}")
            if "AttributeError" in stderr:
                if "get_volatility" in stderr:
                    errors.append("Fake method call: get_volatility() does not exist")
                else:
                    errors.append(f"Attribute error: {stderr}")
            
            return {
                "passed": result["StatusCode"] == 0 and len(errors) == 0,
                "errors": errors,
                "exit_code": result["StatusCode"],
                "stdout": stdout,
                "stderr": stderr
            }
            
        except Exception as e:
            return {
                "passed": False,
                "errors": [f"Container execution failed: {str(e)}"],
                "exit_code": -1
            }
        finally:
            container.remove(force=True)
    
    def _build_validation_image(self, client):
        """Build the validation Docker image"""
        dockerfile = """
FROM python:3.10-slim
RUN pip install numpy pandas scikit-learn
RUN useradd -m validator
USER validator
WORKDIR /code
"""
        client.images.build(
            fileobj=io.BytesIO(dockerfile.encode()),
            tag=self.validation_image
        )
    
    def _analyze_runtime_behavior(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze runtime behavior for deceptive patterns"""
        errors = []
        
        # Check stdout/stderr for suspicious patterns
        output = execution_result.get("stdout", "") + execution_result.get("stderr", "")
        
        # Patterns that indicate deception
        deceptive_patterns = [
            ("calling non-existent method", "Runtime deception detected"),
            ("synthetic data generation", "Data policy violation"),
            ("VIOLATION:", "Compliance violation detected"),
            ("get_volatility", "Suspicious method call pattern")
        ]
        
        for pattern, message in deceptive_patterns:
            if pattern in output:
                errors.append(f"{message}: {pattern} found in output")
        
        return {"passed": len(errors) == 0, "errors": errors}
    
    def _generate_certificate(self, code_path: Path, *results) -> Dict[str, Any]:
        """Generate cryptographically signed certificate of validation"""
        # Combine all validation results
        combined = {
            "code_path": str(code_path),
            "timestamp": time.time(),
            "validations": {
                "static_analysis": results[0],
                "execution": results[1],
                "behavior_analysis": results[2]
            }
        }
        
        # Generate hash of code + results
        code_hash = self._calculate_file_hash(code_path)
        
        certificate = {
            "code_hash": code_hash,
            "validation_hash": hashlib.sha256(
                json.dumps(combined, sort_keys=True).encode()
            ).hexdigest(),
            "valid_until": time.time() + 3600,  # 1 hour validity
            "passed": True,
            "details": combined
        }
        
        # In production, this would be cryptographically signed
        certificate["signature"] = hashlib.sha256(
            (certificate["code_hash"] + certificate["validation_hash"]).encode()
        ).hexdigest()
        
        logger.info(f"Generated validation certificate for {code_path}")
        return certificate
    
    def verify_certificate(self, certificate: Dict[str, Any]) -> bool:
        """Verify a validation certificate"""
        # Check expiry
        if time.time() > certificate.get("valid_until", 0):
            return False
        
        # Verify signature
        expected_sig = hashlib.sha256(
            (certificate["code_hash"] + certificate["validation_hash"]).encode()
        ).hexdigest()
        
        return certificate.get("signature") == expected_sig

# Global validator instance
EXECUTION_VALIDATOR = MandatoryExecutionValidator()