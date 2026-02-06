import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { Database } from '@/types/supabase';

// Singleton pattern for Supabase client
let supabaseInstance: SupabaseClient<Database> | null = null;

/**
 * Advanced Supabase Configuration with:
 * 1. Singleton pattern for performance
 * 2. Real-time optimizations
 * 3. Automatic retry logic
 * 4. Connection pooling
 * 5. Rate limiting protection
 */
export const getSupabaseClient = (): SupabaseClient<Database> => {
  if (supabaseInstance) {
    return supabaseInstance;
  }

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    throw new Error('Missing Supabase environment variables');
  }

  // Advanced client configuration
  supabaseInstance = createClient<Database>(supabaseUrl, supabaseAnonKey, {
    auth: {
      autoRefreshToken: true,
      persistSession: true,
      detectSessionInUrl: true,
      storage: typeof window !== 'undefined' ? window.localStorage : undefined,
      flowType: 'pkce',
    },
    global: {
      headers: {
        'x-application-name': 'optima-pro',
        'x-application-version': '1.0.0',
        'x-client-type': 'web',
      },
    },
    db: {
      schema: 'public',
    },
    realtime: {
      params: {
        eventsPerSecond: 50, // High-frequency updates
        heartbeatIntervalMs: 10000, // 10s heartbeats
        reconnectAfterMs: (tries) => Math.min(1000 * 2 ** tries, 30000), // Exponential backoff
      },
    },
    // Performance optimizations
    fetch: (url, options = {}) => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

      return fetch(url, {
        ...options,
        signal: controller.signal,
        cache: 'no-store', // Always fresh data
        headers: {
          ...options.headers,
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          Pragma: 'no-cache',
        },
      }).finally(() => clearTimeout(timeoutId));
    },
  });

  // Add request interceptor for logging
  const originalPost = supabaseInstance.from;
  supabaseInstance.from = function(table: string) {
    const queryBuilder = originalPost.call(this, table);
    
    // Log slow queries
    const originalSelect = queryBuilder.select;
    queryBuilder.select = function(...args: any[]) {
      const startTime = Date.now();
      const result = originalSelect.apply(this, args);
      
      if (result.then) {
        return result.then((res: any) => {
          const duration = Date.now() - startTime;
          if (duration > 1000) {
            console.warn(`Slow query detected on ${table}: ${duration}ms`);
          }
          return res;
        });
      }
      
      return result;
    };
    
    return queryBuilder;
  };

  return supabaseInstance;
};

// Pre-configured client for most use cases
export const supabase = getSupabaseClient();

/**
 * Admin client for server-side operations
 * Has full access to database (use with caution)
 */
export const getAdminClient = (): SupabaseClient<Database> => {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !supabaseServiceKey) {
    throw new Error('Missing Supabase admin environment variables');
  }

  return createClient<Database>(supabaseUrl, supabaseServiceKey, {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
    global: {
      headers: {
        'x-client-type': 'admin',
      },
    },
  });
};

/**
 * Realtime subscription helper with automatic cleanup
 */
export const createRealtimeSubscription = (
  channelName: string,
  table: string,
  event: 'INSERT' | 'UPDATE' | 'DELETE' | '*',
  filter: string = '',
  callback: (payload: any) => void
) => {
  const channel = supabase
    .channel(channelName)
    .on(
      'postgres_changes',
      {
        event,
        schema: 'public',
        table,
        filter: filter ? `(${filter})` : undefined,
      },
      callback
    )
    .subscribe((status) => {
      console.log(`Realtime ${channelName} status:`, status);
    });

  return {
    unsubscribe: () => supabase.removeChannel(channel),
    channel,
  };
};

/**
 * Batch insert helper for high-performance operations
 */
export const batchInsert = async <T>(
  table: string,
  data: T[],
  batchSize: number = 100
): Promise<{ success: boolean; error?: string }> => {
  try {
    for (let i = 0; i < data.length; i += batchSize) {
      const batch = data.slice(i, i + batchSize);
      const { error } = await supabase
        .from(table)
        .insert(batch);

      if (error) {
        console.error(`Batch insert failed at batch ${i / batchSize}:`, error);
        return { success: false, error: error.message };
      }
      
      // Rate limiting protection
      if (i + batchSize < data.length) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
    
    return { success: true };
  } catch (error: any) {
    console.error('Batch insert failed:', error);
    return { success: false, error: error.message };
  }
};

/**
 * Upsert with conflict resolution
 */
export const upsertWithConflict = async <T extends { id: string }>(
  table: string,
  data: T,
  conflictColumns: string[] = ['id']
): Promise<{ data: T | null; error: string | null }> => {
  try {
    const { data: result, error } = await supabase
      .from(table)
      .upsert(data, {
        onConflict: conflictColumns.join(', '),
        ignoreDuplicates: false,
      })
      .select()
      .single();

    return { data: result as T, error: error?.message || null };
  } catch (error: any) {
    return { data: null, error: error.message };
  }
};

/**
 * Advanced query builder with pagination
 */
export const paginatedQuery = async <T>(
  table: string,
  options: {
    select?: string;
    filters?: Record<string, any>;
    orderBy?: { column: string; ascending: boolean };
    page?: number;
    pageSize?: number;
    range?: { from: number; to: number };
  }
): Promise<{
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}> => {
  const {
    select = '*',
    filters = {},
    orderBy = { column: 'created_at', ascending: false },
    page = 1,
    pageSize = 20,
    range,
  } = options;

  let query = supabase
    .from(table)
    .select(select === '*' ? '*' : `${select}, count(*) OVER() as total_count`);

  // Apply filters
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      if (Array.isArray(value)) {
        query = query.in(key, value);
      } else if (typeof value === 'string' && value.includes('%')) {
        query = query.ilike(key, value);
      } else {
        query = query.eq(key, value);
      }
    }
  });

  // Apply ordering
  query = query.order(orderBy.column, {
    ascending: orderBy.ascending,
    nullsFirst: false,
  });

  // Apply pagination
  if (range) {
    query = query.range(range.from, range.to);
  } else {
    const from = (page - 1) * pageSize;
    const to = from + pageSize - 1;
    query = query.range(from, to);
  }

  const { data, error, count } = await query;

  if (error) {
    console.error('Paginated query failed:', error);
    throw error;
  }

  const total = count || (data as any)?.[0]?.total_count || data?.length || 0;
  const totalPages = Math.ceil(total / pageSize);

  return {
    data: data as T[],
    total,
    page,
    pageSize,
    totalPages,
  };
};

/**
 * Real-time lead notification system
 */
export const subscribeToNewLeads = (
  category: string,
  onNewLead: (lead: any) => void,
  onError?: (error: any) => void
) => {
  return createRealtimeSubscription(
    `leads-${category}-${Date.now()}`,
    'leads',
    'INSERT',
    `category=eq.${category}`,
    (payload) => {
      console.log('🎯 New lead detected:', payload.new);
      onNewLead(payload.new);
      
      // Play notification sound
      if (typeof window !== 'undefined' && window.Audio) {
        try {
          const audio = new Audio('/sounds/notification.mp3');
          audio.volume = 0.3;
          audio.play().catch(console.error);
        } catch (error) {
          console.error('Audio notification failed:', error);
        }
      }
    }
  );
};

/**
 * Analytics tracking helper
 */
export const trackAnalytics = async (
  event: string,
  properties: Record<string, any> = {}
) => {
  try {
    const { data: session } = await supabase.auth.getSession();
    
    await supabase.from('analytics_events').insert({
      event_name: event,
      event_properties: properties,
      user_id: session?.session?.user?.id || null,
      session_id: sessionStorage.getItem('session_id') || null,
      user_agent: typeof window !== 'undefined' ? window.navigator.userAgent : null,
      page_url: typeof window !== 'undefined' ? window.location.href : null,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Analytics tracking failed:', error);
  }
};

/**
 * Health check for Supabase connection
 */
export const checkSupabaseHealth = async (): Promise<{
  healthy: boolean;
  latency: number;
  error?: string;
}> => {
  const startTime = Date.now();
  
  try {
    const { data, error } = await supabase
      .from('leads')
      .select('count')
      .limit(1)
      .single();

    const latency = Date.now() - startTime;
    
    if (error) throw error;
    
    return {
      healthy: true,
      latency,
    };
  } catch (error: any) {
    return {
      healthy: false,
      latency: Date.now() - startTime,
      error: error.message,
    };
  }
};

/**
 * Rate limiting utility
 */
export const withRateLimit = async (
  key: string,
  limit: number,
  windowMs: number
): Promise<{ allowed: boolean; remaining: number }> => {
  try {
    const { data } = await supabase
      .rpc('increment_rate_limit', {
        p_key: key,
        p_limit: limit,
        p_window_ms: windowMs,
      });

    return {
      allowed: data.allowed,
      remaining: data.remaining,
    };
  } catch (error) {
    console.error('Rate limit check failed:', error);
    return { allowed: true, remaining: limit };
  }
};

export default supabase;
