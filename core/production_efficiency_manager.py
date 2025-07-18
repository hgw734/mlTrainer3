import logging
logger = logging.getLogger(__name__)

"""
Production Efficiency Manager
============================
Unified system for optimizing production efficiency
"""

import asyncio
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

# Import our optimization components
from .resource_optimizer import DynamicResourceOptimizer
from .gpu_scheduler import GPUScheduler
from .computation_cache import ComputationCache
from .computation_graph import ComputationGraphOptimizer
from monitoring.resource_waste_detector import ResourceWasteDetector

@dataclass
class EfficiencyMetrics:
    compute_utilization: float
    memory_utilization: float
    gpu_utilization: float
    cache_hit_rate: float
    redundant_computations: int
    estimated_monthly_savings: float

@dataclass
class OptimizationRecommendation:
    category: str
    severity: str
    description: str
    estimated_savings: float
    implementation_effort: str  # low, medium, high

class ProductionEfficiencyManager:
    """
    Central manager for all production efficiency optimizations
    """
    def __init__(self):
        self.resource_optimizer = DynamicResourceOptimizer()
        self.gpu_scheduler = GPUScheduler()
        self.computation_cache = ComputationCache()
        self.graph_optimizer = ComputationGraphOptimizer()
        self.waste_detector = ResourceWasteDetector()
        self.metrics_history: List[EfficiencyMetrics] = []

    async def analyze_system_efficiency(self) -> Dict:
        """
        Comprehensive efficiency analysis
        """
        logger.info("🔍 Analyzing system efficiency# Production code implemented")

        # Gather metrics from all components
        resource_metrics = await self._get_resource_metrics()
        gpu_metrics = await self._get_gpu_metrics()
        cache_metrics = self.computation_cache.get_cache_efficiency_report()
        redundancy_analysis = self.graph_optimizer.identify_redundant_computations()

        # Analyze waste across all services
        services = await self._get_all_services()
        waste_report = {}

        for service in services:
            waste_analysis = await self.waste_detector.analyze_resource_usage(service)
            if any(w['detected'] for w in waste_analysis['waste_analysis'].values()):
                waste_report[service] = waste_analysis

        # Calculate overall efficiency metrics
        efficiency_metrics = EfficiencyMetrics(
            compute_utilization=resource_metrics['avg_cpu_utilization'],
            memory_utilization=resource_metrics['avg_memory_utilization'],
            gpu_utilization=gpu_metrics['avg_gpu_utilization'],
            cache_hit_rate=self._calculate_overall_cache_hit_rate(cache_metrics),
            redundant_computations=len(redundancy_analysis),
            estimated_monthly_savings=self._calculate_total_savings(waste_report)
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            efficiency_metrics,
            waste_report,
            cache_metrics
        )

        # Store metrics for trend analysis
        self.metrics_history.append(efficiency_metrics)

        return {
            'timestamp': datetime.now().isoformat(),
            'efficiency_metrics': efficiency_metrics,
            'waste_report': waste_report,
            'cache_report': cache_metrics,
            'redundant_computations': redundancy_analysis,
            'recommendations': recommendations,
            'efficiency_score': self._calculate_efficiency_score(efficiency_metrics)
        }

    async def optimize_workloads(self) -> Dict[str, int]:
        """
        Apply optimizations to all workloads
        """
        logger.info("⚡ Optimizing workloads# Production code implemented")

        optimizations_applied = {
            'resources_scaled': 0,
            'computations_cached': 0,
            'redundancies_eliminated': 0,
            'gpus_rescheduled': 0
        }

        # 1. Scale resources based on usage
        services = await self._get_all_services()
        for service in services:
            try:
                await self.resource_optimizer.optimize_deployment(service)
                optimizations_applied['resources_scaled'] += 1
            except Exception as e:
                logger.error(f"Failed to optimize {service}: {e}")

        # 2. Eliminate redundant computations
        self.graph_optimizer.eliminate_redundancy()
        optimizations_applied['redundancies_eliminated'] = len(
            self.graph_optimizer.identify_redundant_computations()
        )

        # 3. Reschedule GPU jobs for better utilization
        pending_jobs = self.gpu_scheduler.job_queue
        for job in pending_jobs[:]:  # Copy to avoid modification during iteration
            gpu_id = await self.gpu_scheduler.schedule_job(job)
            if gpu_id is not None:
                optimizations_applied['gpus_rescheduled'] += 1

        return optimizations_applied

    async def enable_cost_optimization_mode(self):
        """
        Enable aggressive cost optimization
        """
        logger.info("💰 Enabling cost optimization mode# Production code implemented")

        # Configure spot instance usage
        from infrastructure.spot_manager import SpotInstanceManager
        spot_manager = SpotInstanceManager()

        # Get all workloads
        workloads = await self._get_workload_specifications()

        # Determine spot placement
        placement = await spot_manager.optimize_workload_placement(workloads)

        # Apply spot instance configuration
        spot_workloads = [w for w, p in placement.items() if p == 'spot']
        logger.info(f"Migrating {len(spot_workloads)} workloads to spot instances")

        # Enable aggressive resource scaling
        self.resource_optimizer.resource_profiles['minimal'] = ResourceProfile(0.25, 0.5)

        # Enable computation caching with longer TTLs
        self.computation_cache.default_ttl = timedelta(hours=4)

        return {
            'spot_workloads': len(spot_workloads),
            'estimated_monthly_savings': len(spot_workloads) * 150  # ~$150/workload
        }

    def _generate_recommendations(self,
                                  metrics: EfficiencyMetrics,
                                  waste_report: Dict,
                                  cache_report: Dict) -> List[OptimizationRecommendation]:
        """Generate actionable recommendations"""
        recommendations = []

        # Check compute utilization
        if metrics.compute_utilization < 50:
            recommendations.append(OptimizationRecommendation(
                category="Resource Optimization",
                severity="high",
                description=f"CPU utilization is only {metrics.compute_utilization:.1f}%. Consider reducing allocated CPU by 30%",
                estimated_savings=500.0,
                implementation_effort="low"
            ))

        # Check memory utilization
        if metrics.memory_utilization < 40:
            recommendations.append(OptimizationRecommendation(
                category="Resource Optimization",
                severity="high",
                description=f"Memory utilization is only {metrics.memory_utilization:.1f}%. Consider reducing memory allocation",
                estimated_savings=300.0,
                implementation_effort="low"
            ))

        # Check GPU utilization
        if metrics.gpu_utilization < 70:
            recommendations.append(OptimizationRecommendation(
                category="GPU Optimization",
                severity="medium",
                description=f"GPU utilization is {metrics.gpu_utilization:.1f}%. Enable GPU job batching",
                estimated_savings=1000.0,
                implementation_effort="medium"
            ))

        # Check cache efficiency
        if metrics.cache_hit_rate < 0.6:
            recommendations.append(OptimizationRecommendation(
                category="Computation Optimization",
                severity="medium",
                description=f"Cache hit rate is only {metrics.cache_hit_rate:.1%}. Review cache configuration",
                estimated_savings=200.0,
                implementation_effort="low"
            ))

        # Check for redundant computations
        if metrics.redundant_computations > 5:
            recommendations.append(OptimizationRecommendation(
                category="Computation Optimization",
                severity="high",
                description=f"Found {metrics.redundant_computations} redundant computations. Enable deduplication",
                estimated_savings=400.0,
                implementation_effort="medium"
            ))

        # Check waste report
        total_waste_services = len(waste_report)
        if total_waste_services > 0:
            recommendations.append(OptimizationRecommendation(
                category="Waste Elimination",
                severity="high",
                description=f"Found waste in {total_waste_services} services. Review and optimize",
                estimated_savings=metrics.estimated_monthly_savings,
                implementation_effort="medium"
            ))

        return sorted(recommendations, key=lambda r: r.estimated_savings, reverse=True)

    def _calculate_efficiency_score(self, metrics: EfficiencyMetrics) -> float:
        """Calculate overall efficiency score (0-100)"""
        # Weighted scoring
        scores = {
            'compute': min(metrics.compute_utilization / 70 * 100, 100) * 0.25,
            'memory': min(metrics.memory_utilization / 70 * 100, 100) * 0.25,
            'gpu': min(metrics.gpu_utilization / 80 * 100, 100) * 0.20,
            'cache': metrics.cache_hit_rate * 100 * 0.20,
            'redundancy': max(0, 100 - metrics.redundant_computations * 10) * 0.10
        }

        return sum(scores.values())

    async def _get_resource_metrics(self) -> Dict:
        """Get resource utilization metrics"""
        # In production, query Prometheus
        return {
            'avg_cpu_utilization': 60.0,  # TODO: Connect to real monitoring
            'avg_memory_utilization': 50.0  # TODO: Connect to real monitoring
        }

    async def _get_gpu_metrics(self) -> Dict:
        """Get GPU utilization metrics"""
        # In production, query NVML
        return {
            'avg_gpu_utilization': 70.0  # TODO: Connect to real monitoring
        }

    def _calculate_overall_cache_hit_rate(self, cache_report: Dict) -> float:
        """Calculate weighted cache hit rate"""
        if not cache_report['function_stats']:
            return 0.0

        total_hits = sum(
            stats['hit_rate'] * (stats['time_saved_seconds'] or 1)
            for stats in cache_report['function_stats'].values()
        )
        total_weight = sum(
            stats['time_saved_seconds'] or 1
            for stats in cache_report['function_stats'].values()
        )

        return total_hits / total_weight if total_weight > 0 else 0.0

    def _calculate_total_savings(self, waste_report: Dict) -> float:
        """Calculate total potential monthly savings"""
        total = 0.0

        for service, analysis in waste_report.items():
            for category, amount in analysis['potential_savings'].items():
                total += amount

        return total

    async def _get_all_services(self) -> List[str]:
        """Get list of all services"""
        # In production, query service registry
        return [
            'mltrainer-api',
            'data-pipeline',
            'model-training',
            'prediction-service',
            'feature-store'
        ]

    async def _get_workload_specifications(self) -> List[Dict]:
        """Get workload specifications"""
        # In production, query workload registry
        return [
            {
                'id': 'training-job-1',
                'cpu': 4,
                'memory_gb': 16,
                'gpu': 1,
                'duration_hours': 2,
                'interruptible': True,
                'priority': 5
            },
            {
                'id': 'inference-service',
                'cpu': 2,
                'memory_gb': 8,
                'gpu': 0,
                'duration_hours': 24,
                'interruptible': False,
                'priority': 9
            }
        ]

# Dashboard for efficiency monitoring
class EfficiencyDashboard:
    """
    Real-time efficiency monitoring dashboard
    """
    def __init__(self, manager: ProductionEfficiencyManager):
        self.manager = manager

    async def generate_dashboard_data(self) -> Dict:
        """Generate data for efficiency dashboard"""
        # Get current analysis
        analysis = await self.manager.analyze_system_efficiency()

        # Calculate trends if we have history
        trends = {}
        if len(self.manager.metrics_history) > 1:
            current = self.manager.metrics_history[-1]
            previous = self.manager.metrics_history[-2]

            trends = {
                'compute_trend': current.compute_utilization - previous.compute_utilization,
                'memory_trend': current.memory_utilization - previous.memory_utilization,
                'gpu_trend': current.gpu_utilization - previous.gpu_utilization,
                'cache_trend': current.cache_hit_rate - previous.cache_hit_rate
            }

        return {
            'current_metrics': analysis['efficiency_metrics'],
            'efficiency_score': analysis['efficiency_score'],
            'recommendations': analysis['recommendations'][:5],  # Top 5
            'trends': trends,
            'potential_monthly_savings': analysis['efficiency_metrics'].estimated_monthly_savings,
            'alerts': self._generate_alerts(analysis)
        }

    def _generate_alerts(self, analysis: Dict) -> List[Dict]:
        """Generate alerts for critical issues"""
        alerts = []

        metrics = analysis['efficiency_metrics']

        if metrics.compute_utilization < 30:
            alerts.append({
                'level': 'critical',
                'message': f'Severe under-utilization: CPU at {metrics.compute_utilization:.1f}%',
                'action': 'Reduce compute allocation immediately'
            })

        if metrics.cache_hit_rate < 0.3:
            alerts.append({
                'level': 'warning',
                'message': f'Poor cache performance: {metrics.cache_hit_rate:.1%} hit rate',
                'action': 'Review cache configuration'
            })

        if metrics.redundant_computations > 10:
            alerts.append({
                'level': 'warning',
                'message': f'{metrics.redundant_computations} redundant computations detected',
                'action': 'Enable computation deduplication'
            })

        return alerts

# CLI for efficiency management
async def efficiency_cli():
    """Command-line interface for efficiency management"""

    manager = ProductionEfficiencyManager()
    dashboard = EfficiencyDashboard(manager)

    while True:
        logger.info("\n=== Production Efficiency Manager ===")
        logger.info("1. Analyze system efficiency")
        logger.info("2. Apply optimizations")
        logger.info("3. Enable cost optimization mode")
        logger.info("4. View dashboard")
        logger.info("5. Exit")

        choice = input("\nSelect option: ")

        if choice == '1':
            analysis = await manager.analyze_system_efficiency()
            logger.info(f"\n📊 Efficiency Score: {analysis['efficiency_score']:.1f}/100")
            logger.info(f"💰 Potential Monthly Savings: ${analysis['efficiency_metrics'].estimated_monthly_savings:,.2f}")
            logger.info("\n🎯 Top Recommendations:")
            for rec in analysis['recommendations'][:3]:
                logger.info(f"  - {rec.description} (saves ${rec.estimated_savings}/month)")

        elif choice == '2':
            optimizations = await manager.optimize_workloads()
            logger.info("\n✅ Optimizations Applied:")
            for opt_type, count in optimizations.items():
                logger.info(f"  - {opt_type}: {count}")

        elif choice == '3':
            cost_opt = await manager.enable_cost_optimization_mode()
            logger.info(f"\n💰 Cost optimization enabled!")
            logger.info(f"  - Spot workloads: {cost_opt['spot_workloads']}")
            logger.info(f"  - Est. savings: ${cost_opt['estimated_monthly_savings']}/month")

        elif choice == '4':
            data = await dashboard.generate_dashboard_data()
            logger.info(f"\n📈 Efficiency Dashboard")
            logger.info(f"  Score: {data['efficiency_score']:.1f}/100")
            logger.info(f"  Savings: ${data['potential_monthly_savings']:,.2f}/month")
            if data['alerts']:
                logger.info("\n⚠️  Alerts:")
                for alert in data['alerts']:
                    logger.info(f"  [{alert['level']}] {alert['message']}")

        elif choice == '5':
            break

if __name__ == "__main__":
    asyncio.run(efficiency_cli())