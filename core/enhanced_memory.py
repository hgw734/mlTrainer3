"""
Enhanced Memory System
======================

Wrapper for unified memory system with enhanced features.
"""

from utils.unified_memory import get_unified_memory


class EnhancedMemoryManager:
    """Enhanced memory manager wrapping unified memory"""

    def __init__(self):
        self.unified_memory = get_unified_memory()
        self.short_term_memory = {}
        self.long_term_memory = {}
        self.episodic_memory = []

        def add_interaction(self, user_role, user_content, assistant_role, assistant_content, metadata=None):
            """Add an interaction to memory"""
            # Add to unified memory
            self.unified_memory.add_message(user_role, user_content)
            self.unified_memory.add_message(assistant_role, assistant_content)

            # Store in short-term memory
            interaction_id = f"interaction_{len(self.short_term_memory)}"
            self.short_term_memory[interaction_id] = {
            "user": {"role": user_role, "content": user_content},
            "assistant": {"role": assistant_role, "content": assistant_content},
            "metadata": metadata or {},
            "importance": 0.5,
            }

            return interaction_id

            def get_recent_context(self, limit=10):
                """Get recent context from memory"""
                return self.unified_memory.get_recent_context(limit)

                def search_memories(self, query, limit=10):
                    """Search memories by query"""
                    return self.unified_memory.search_by_topic(query)[:limit]

                    def get_memory_stats(self):
                        """Get memory statistics"""
                        base_stats = self.unified_memory.get_memory_stats()
                        base_stats.update(
                        {
                        "short_term_count": len(self.short_term_memory),
                        "long_term_count": len(self.long_term_memory),
                        "episodic_count": len(self.episodic_memory),
                        }
                        )
                        return base_stats


                        # Singleton instance
                        _memory_manager = None


                        def get_memory_manager():
                            """Get the enhanced memory manager instance"""
                            global _memory_manager
                            if _memory_manager is None:
                                _memory_manager = EnhancedMemoryManager()
                                return _memory_manager
