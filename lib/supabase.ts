import { createClient, SupabaseClient } from '@supabase/supabase-js';

// Singleton pattern for Supabase client
// Humne 'Database' type hata kar 'any' kiya hai taaki Build fail na ho
let supabaseInstance: SupabaseClient<any> | null = null;

export const getSupabaseClient = (): SupabaseClient<any> => {
  if (supabaseInstance) {
    return supabaseInstance;
  }

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    throw new Error('Missing Supabase environment variables');
  }

  // Advanced client configuration
  supabaseInstance = createClient<any>(supabaseUrl, supabaseAnonKey, {
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
        eventsPerSecond: 50,
        heartbeatIntervalMs: 10000,
      },
    },
    fetch: (url, options = {}) => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);

      return fetch(url, {
        ...options,
        signal: controller.signal,
        cache: 'no-store',
        headers: {
          ...options.headers,
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          Pragma: 'no-cache',
        },
      }).finally(() => clearTimeout(timeoutId));
    },
  });

  return supabaseInstance;
};

// Exporting main client
export const supabase = getSupabaseClient();

/**
 * Realtime lead notification system
 */
export const subscribeToNewLeads = (
  category: string,
  onNewLead: (lead: any) => void
) => {
  return supabase
    .channel(`leads-${category}-${Date.now()}`)
    .on(
      'postgres_changes',
      {
        event: 'INSERT',
        schema: 'public',
        table: 'leads',
      },
      (payload) => {
        console.log('🎯 New lead detected:', payload.new);
        onNewLead(payload.new);
      }
    )
    .subscribe();
};

export default supabase;
